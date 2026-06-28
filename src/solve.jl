
module Solving

# Constants for the new producer-consumer logic
const INTERSECTION_GENERATION_CHUNK_SIZE = 10000    # Number of simplices each generator thread processes at once
const SOLVER_UPDATE_THRESHOLD = 500000              # Number of new clauses to buffer before updating the solver
                                                    # Only relevant when using incremental solving

using PicoSAT

# Include the Cadical wrapper
include("CadicalWrapper.jl")
using .CadicalWrapper

#= No longer using d4
include("d4Wrapper.jl")
using .D4AllSat
=# 
include("subdivision_regularity.jl")
using .SubdivisionRegularity

using ..Structs
using ..Helpers
using ..BasicComputations
using Base.Threads

export solve_picosat, solve_cadical_incremental, solve_cadical_standard, solve_parallel#, find_all_d4 # No longer using d4

function solve_picosat(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool, stop_signal::Threads.Atomic{Bool})
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0
    number_of_flag_triangulations_found = 0
    number_of_quadratic_triangulations_found = 0

    for solution in PicoSAT.itersolve(cnf)
        # Check if another thread has already signaled to stop
        if stop_signal[]
            break
        end

        if number_of_triangulations_found % 1000 == 0 && number_of_triangulations_found > 0 && show_running_updates
            ghost_print(" ($number_of_triangulations_found triangulations found)")
        end
        
        sol_indices = findall(l -> l > 0, solution)
        simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in sol_indices]
        
        number_of_triangulations_found += 1
        
        if isempty(first_solution_simplices)
            first_solution_simplices = simplices
        end

        should_terminate = false

        # --- Logic gates for config settings w.r.t regularity, flags, and termination ---
        if !config.regular
            if config.flag_triangulation
                if is_flag_triangulation(simplices)
                    number_of_flag_triangulations_found += 1
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all; should_terminate = true; end
                elseif show_running_updates
                    ghost_print(" ($number_of_triangulations_found non-flag triangulations found)")
                end
            else # !config.flag_triangulation
                if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                    push!(solution_simplices, simplices)
                end
                if !config.find_all; should_terminate = true; end
            end
        else # config.regular == true
            if is_regular(simplices)
                number_of_regular_triangulations_found += 1
                
                if config.flag_triangulation # Looking for quadratic (regular + flag) triangulations
                    if is_flag_triangulation(simplices)
                        number_of_quadratic_triangulations_found += 1
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                            push!(solution_simplices, simplices)
                        end
                        if !config.find_all; should_terminate = true; end
                    elseif show_running_updates
                        ghost_print(" ($number_of_triangulations_found regular non-flag triangulations found)")
                    end
                else # Regular, but not filtering for flags
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all; should_terminate = true; end
                end
            elseif show_running_updates
                ghost_print(" ($number_of_triangulations_found non-regular triangulations found)")
            end
        end

        if should_terminate
            Threads.atomic_cas!(stop_signal, false, true)
            break
        end
    end

    return (
        solution_simplices, 
        first_solution_simplices, 
        number_of_triangulations_found, 
        number_of_regular_triangulations_found,
        number_of_flag_triangulations_found,
        number_of_quadratic_triangulations_found
    )
end

function solve_cadical_incremental(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, dim::Int, config::Config, show_running_updates::Bool, log_verbose::Function)
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0
    num_simplices = length(S_indices)

    # --- Producer-Consumer Incremental Logic ---
    solver = CadicalWrapper.Solver()

    # Add initial clauses
    for clause in cnf
        CadicalWrapper.add_clause(solver, clause)
    end

    log_verbose("      (Async) Calculating intersection pairs incrementally and feeding to solver...")

    # 1. Prepare Optimized Simplex Data
    cpu_simplices = CPUIntersection.prepare_simplices_cpu(P, S_indices, Val(dim))

    # 2. Shared State for Intersection Generators
    clause_channel = Channel{Vector{Int}}(100000)
    next_simplex_idx = Threads.Atomic{Int}(1)
    generation_complete = Threads.Atomic{Bool}(false)
    
    # Error flag to detect generator failures
    generator_failed = Threads.Atomic{Bool}(false)

    # 3. Launch Generator Threads with Error Handling
    num_workers = max(1, nthreads() - 1)
    generator_tasks = []

    for t_id in 1:num_workers
        t = Threads.@spawn begin
            try
                while true
                    # Grab a chunk of work
                    i = Threads.atomic_add!(next_simplex_idx, 1)
                    if i >= num_simplices
                        break
                    end
                    
                    # Generate conflicts for this simplex against all subsequent ones
                    conflicts = CPUIntersection.check_intersections_range_cpu(cpu_simplices, i, i+1, num_simplices)
                    
                    for c in conflicts
                        put!(clause_channel, c)
                    end
                end
            catch e
                # LOG ERROR and set failure flag
                @error "Generator thread $t_id failed: $e"
                Base.show_backtrace(stderr, catch_backtrace())
                generator_failed[] = true
                # Close channel to unblock consumer if needed (optional but risky if others are running)
            end
        end
        push!(generator_tasks, t)
    end

    # Monitor task to close channel when all generators are done
    Threads.@spawn begin
        try
            for t in generator_tasks
                wait(t)
            end
        catch e
            @error "Monitor thread error waiting for generators: $e"
            generator_failed[] = true
        finally
            # Always signal completion, even if failed, so the main loop doesn't hang
            generation_complete[] = true
        end
    end

    # 4. Start Solver Async
    task = CadicalWrapper.solve_async(solver)
    number_of_clauses_added = 0
    new_clauses_buffer = Vector{Vector{Int}}()

    # 5. Coordinator Loop
    while true
        # Check for generator failure
        if generator_failed[]
            @error "Aborting solving due to generator failure."
            break
        end

        # --- A. Check Solver Status ---
        if istaskdone(task)
            res = fetch(task)
            
            if res == 10 # SAT
                # Retrieve candidate solution
                solution_vector = Int[]
                for i in 1:num_simplices
                    if CadicalWrapper.val(solver, i) > 0
                        push!(solution_vector, i)
                    end
                end
                
                simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in solution_vector]
                number_of_triangulations_found += 1
                
                if isempty(first_solution_simplices)
                    first_solution_simplices = simplices
                end

                should_terminate = false

                if !config.regular
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all
                        should_terminate = true
                    end
                else
                    if is_regular(simplices)
                        if isempty(first_regular_solution_simplices)
                            first_regular_solution_simplices = simplices
                        end
                        number_of_regular_triangulations_found += 1
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                            push!(solution_simplices, simplices)
                        end
                        if !config.find_all
                            should_terminate = true
                        end
                    else
                        if show_running_updates
                            ghost_print(" ($number_of_triangulations_found non-regular triangulations found)")
                        end
                        # Valid but not regular -> continue search
                    end
                end

                if should_terminate
                    break
                else
                    # Block this solution and resume
                    CadicalWrapper.add_clause(solver, [-solution_vector...])
                    task = CadicalWrapper.solve_async(solver)
                end
                
            elseif res == 20 # UNSAT
                # If generators are done, it's definitively UNSAT.
                break
            else
                # Unknown/Interrupted - Just resume
                task = CadicalWrapper.solve_async(solver)
            end
        end

        # --- B. Process Incoming Clauses ---
        # Drain channel into local buffer, BUT LIMIT IT to prevent starvation
        # We fetch up to Update Threshold + a bit, then force a check/update cycle
        clauses_fetched_this_cycle = 0
        limit = SOLVER_UPDATE_THRESHOLD * 2 
        
        while isready(clause_channel) && clauses_fetched_this_cycle < limit
            try
                push!(new_clauses_buffer, take!(clause_channel))
                clauses_fetched_this_cycle += 1
            catch
                break
            end
        end

        # --- C. Update Solver if Threshold Reached ---
        should_update = length(new_clauses_buffer) >= SOLVER_UPDATE_THRESHOLD ||
                        (generation_complete[] && !isempty(new_clauses_buffer))
        
        # Only interrupt if the task is NOT done. If it is done, we loop around to 'A' to handle result first.
        if should_update && !istaskdone(task)
            number_of_clauses_added += length(new_clauses_buffer)
            ghost_print("   Number of clauses added: $number_of_clauses_added")
            
            CadicalWrapper.interrupt(solver)
            
            # Wait for task to effectively stop
            try
                wait(task)
            catch
            end
            
            # CRITICAL CHECK: Did the solver finish with a result (SAT/UNSAT) while we tried to interrupt?
            # If so, we MUST process that result before adding clauses, otherwise we might corrupt the state
            # or miss a solution.
            if istaskdone(task)
                # If done, we loop back to A immediately to process the result.
                # We do NOT add clauses yet. The buffer remains for the next pass.
                continue 
            end

            # Add buffered clauses
            for c in new_clauses_buffer
                CadicalWrapper.add_clause(solver, c)
            end
            empty!(new_clauses_buffer)
            
            # Resume solver
            task = CadicalWrapper.solve_async(solver)
        end
    end
    
    CadicalWrapper.release(solver)
    
    return solution_simplices, first_solution_simplices, first_regular_solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found
end

function solve_cadical_standard(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool,
                                stop_signal::Threads.Atomic{Bool})
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0
    number_of_flag_triangulations_found = 0
    number_of_quadratic_triangulations_found = 0
    num_simplices = length(S_indices)

    solver = CadicalWrapper.Solver()
    for clause in cnf
        CadicalWrapper.add_clause(solver, clause)
    end

    while true
        if stop_signal[]
            break
        end
        if number_of_triangulations_found % 1000 == 0 && number_of_triangulations_found > 0 && show_running_updates
            ghost_print(" ($number_of_triangulations_found triangulations found)")
        end
        
        res = CadicalWrapper.solve_blocking(solver)
        if res == 10 # SAT
            # Retrieve solution
            solution_vector = Int[]
            for i in 1:num_simplices
                if CadicalWrapper.val(solver, i) > 0
                    push!(solution_vector, i)
                end
            end
            simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in solution_vector]
            
            number_of_triangulations_found += 1
            if isempty(first_solution_simplices)
                first_solution_simplices = simplices
            end

            should_terminate = false

            # --- Logic gates for config settings w.r.t regularity, flags, and termination ---
            if !config.regular
                if config.flag_triangulation
                    if is_flag_triangulation(simplices)
                        number_of_flag_triangulations_found += 1
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                            push!(solution_simplices, simplices)
                        end
                        if !config.find_all; should_terminate = true; end
                    elseif show_running_updates
                        ghost_print(" ($number_of_triangulations_found non-flag triangulations found)")
                    end
                else # !config.flag_triangulation
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all; should_terminate = true; end
                end
            else # config.regular == true
                if is_regular(simplices)
                    if isempty(first_solution_simplices)
                        first_solution_simplices = simplices
                    end
                    number_of_regular_triangulations_found += 1
                    
                    if config.flag_triangulation # Looking for quadratic (regular + flag) triangulations
                        if is_flag_triangulation(simplices)
                            number_of_quadratic_triangulations_found += 1
                            if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                                push!(solution_simplices, simplices)
                            end
                            if !config.find_all; should_terminate = true; end
                        elseif show_running_updates
                            ghost_print(" ($number_of_triangulations_found regular non-flag triangulations found)")
                        end
                    else # Regular, but not filtering for flags
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                            push!(solution_simplices, simplices)
                        end
                        if !config.find_all; should_terminate = true; end
                    end
                elseif show_running_updates
                    ghost_print(" ($number_of_triangulations_found non-regular triangulations found)")
                end
            end

            if should_terminate
                Threads.atomic_cas!(stop_signal, false, true)
                break
            else
                CadicalWrapper.add_clause(solver, [-solution_vector...])
            end
        else # UNSAT or Unknown
            break
        end
    end
    CadicalWrapper.release(solver)

    return (
        solution_simplices, 
        first_solution_simplices, 
        number_of_triangulations_found, 
        number_of_regular_triangulations_found, 
        number_of_flag_triangulations_found, 
        number_of_quadratic_triangulations_found
    )
end

function solve_parallel(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool)
    num_threads = Threads.nthreads()

    # 1. find generic point and central simplices
    generic_point = find_generic_point(P, internal_faces(P, size(P, 2)), Val(size(P, 2)))
    central_indices_map = compute_central_indices(P, S_indices, generic_point)
    
    solve_function = nothing
    if config.solver == "cadical"
        solve_function = solve_cadical_standard
    elseif config.solver == "picosat"
        solve_function = solve_picosat
    else
        error("Unsupported solver for parallel execution: $(config.solver)")
    end

    # 2. split central simplices into groups for each thread (round-robin)
    central_groups = [Int[] for _ in 1:num_threads]
    for (i, idx) in enumerate(central_indices_map)
        push!(central_groups[mod1(i, num_threads)], idx)
    end

    # 3. prepare thread-local storage for results
    solution_simplices_threads = [Vector{Vector{Matrix{Int}}}() for _ in 1:num_threads]
    first_solution_threads     = [Vector{Matrix{Int}}() for _ in 1:num_threads]
    
    # Thread-isolated quantitative tracking
    num_triangulations_threads = zeros(Int, num_threads)
    num_regular_threads        = zeros(Int, num_threads)
    num_flag_threads           = zeros(Int, num_threads)
    num_quadratic_threads      = zeros(Int, num_threads)
    
    # Shared stop signal for early termination across all threads
    found_solution = Threads.Atomic{Bool}(false)

    # 4. thread i solves cnf + [central_group[i]...] using the standard solver
    # This limits thread i to solutions where a simplex from its central group is used.
    Threads.@threads for tid in 1:num_threads
        group = central_groups[tid]
        if isempty(group)
            continue
        end

        # Create a shallow copy of the CNF and add this thread's group constraint
        local_cnf = copy(cnf)
        push!(local_cnf, group)

        # Call the standard solving logic, suppressing live updates to avoid thread clutter
        sol_simp, first_sol, num_found, num_reg, num_flag, num_quad = solve_function(
            local_cnf, P, S_indices, config, false, found_solution
        )

        # Store the returned values into the isolated thread arrays
        solution_simplices_threads[tid] = sol_simp
        first_solution_threads[tid]     = first_sol
        
        num_triangulations_threads[tid] = num_found
        num_regular_threads[tid]        = num_reg
        num_flag_threads[tid]           = num_flag
        num_quadratic_threads[tid]      = num_quad
    end

    # 5. aggregate the results from all threads
    solution_simplices       = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    
    # Compute totals
    number_of_triangulations_found           = sum(num_triangulations_threads)
    number_of_regular_triangulations_found   = sum(num_regular_threads)
    number_of_flag_triangulations_found      = sum(num_flag_threads)
    number_of_quadratic_triangulations_found = sum(num_quadratic_threads)

    for tid in 1:num_threads
        append!(solution_simplices, solution_simplices_threads[tid])
        
        # Capture the very first valid solution found across threads
        if isempty(first_solution_simplices) && !isempty(first_solution_threads[tid])
            first_solution_simplices = first_solution_threads[tid]
        end
    end

    # If only the first triangulation is requested, truncate the combined list
    if config.return_triangulations == "first" && length(solution_simplices) > 1
        resize!(solution_simplices, 1)
    end

    # Print a final summary if requested
    if show_running_updates
        ghost_print("Parallel search finished: $number_of_triangulations_found triangulations found " *
                    "($number_of_regular_triangulations_found regular, " *
                    "$number_of_flag_triangulations_found flag, " *
                    "$number_of_quadratic_triangulations_found quadratic).")
    end

    return (
        solution_simplices, 
        first_solution_simplices, 
        number_of_triangulations_found, 
        number_of_regular_triangulations_found,
        number_of_flag_triangulations_found,
        number_of_quadratic_triangulations_found
    )
end

#= No longer using d4
function find_all_d4(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config, show_running_updates::Bool)
    num_threads = Threads.nthreads()

    # 1. Find generic point and central simplices
    generic_point = find_generic_point(P, internal_faces(P, size(P, 2)), Val(size(P, 2)))
    central_indices_map = compute_central_indices(P, S_indices, generic_point)

    # 2. Split central simplices into groups for each thread (round-robin)
    central_groups = [Int[] for _ in 1:num_threads]
    for (i, idx) in enumerate(central_indices_map)
        push!(central_groups[mod1(i, num_threads)], idx)
    end

    # 3. Standard thread-agnostic state variables
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0

    # 4. Consume the safely merged stream of solutions
    for solution in d4_itersolve(cnf, central_groups)
        if number_of_triangulations_found % 1000 == 0 && number_of_triangulations_found > 0 && show_running_updates
            ghost_print(" ($number_of_triangulations_found triangulations found)")
        end
        
        sol_indices = findall(l -> l > 0, solution)
        simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in sol_indices]
        number_of_triangulations_found += 1
        
        if isempty(first_solution_simplices)
            first_solution_simplices = simplices
        end
        
        if !config.regular
            if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                push!(solution_simplices, simplices)
            end
        else
            if is_regular(simplices)
                if isempty(first_regular_solution_simplices)
                    first_regular_solution_simplices = simplices
                end
                number_of_regular_triangulations_found += 1
                if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                    push!(solution_simplices, simplices)
                end
            elseif show_running_updates
                ghost_print(" ($number_of_triangulations_found non-regular triangulations found)")
            end
        end
    end
    
    return solution_simplices, first_solution_simplices, first_regular_solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found
end
=#
end

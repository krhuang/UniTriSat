module Solvers

using ..Structs

export solve_picosat, solve_cadical_incremental, solve_cadical_standard, solve_cadical_parallel

function solve_picosat(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool)
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0

    for solution in PicoSAT.itersolve(cnf)
        s = " ($number_of_triangulations_found triangulations found)"
        print(s*"\b"^(length(s)))
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
            if !config.find_all; break; end
        else
            if is_regular(simplices)
                if isempty(first_regular_solution_simplices)
                    first_regular_solution_simplices = simplices
                end
                number_of_regular_triangulations_found += 1
                if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                    push!(solution_simplices, simplices)
                end
                if !config.find_all; break; end
            elseif show_running_updates
                s = " ($number_of_triangulations_found non-regular triangulations found)"
                print(s*"\b"^(length(s)))
            end
        end
    end
    return solution_simplices, first_solution_simplices, first_regular_solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found
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
                            s = " ($number_of_triangulations_found non-regular triangulations found)"
                            print(s*"\b"^(length(s)))
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
            s = "   Number of clauses added: $number_of_clauses_added"
            print("$s"*"\b"^length(s))
            
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

function solve_cadical_standard(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool)
    println("entering cadical standard function?")
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}() # Recording the first solution of the desired type (flag, regular)

    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0
    number_of_flag_triangulations_found = 0
    number_of_quadratic_triangulations_found = 0
    # number_of_non_flag_regular_triangulations_found = 0 The user can easily compute this number from the previous infos given
    num_simplices = length(S_indices)

    solver = CadicalWrapper.Solver()
    for clause in cnf
        CadicalWrapper.add_clause(solver, clause)
    end

    while true
        if number_of_triangulations_found%1000 == 0 && show_running_updates
            s = " ($number_of_triangulations_found triangulations found...)"
            print(s*"\b"^(length(s)))
        end
        res = CadicalWrapper.solve_blocking(solver)
        if res == 10 # SATISFIABLE
            # Retrieve solution
            solution_vector = Int[]
            for i in 1:num_simplices
                if CadicalWrapper.val(solver, i) > 0
                    push!(solution_vector, i)
                end
            end
            simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in solution_vector] # Convert solution vector into a triangulation
            
            number_of_triangulations_found += 1

            should_terminate = false

            # Logic gates for config settings w.r.t. regularity, flag, find_all

            if !config.regular 
                if config.flag_triangulation
                    if is_flag_triangulation(simplices) # Check if solution is flag
                        if isempty(first_flag_solution_simplices)
                            first_solution_simplices = simplices 
                        end 
                        number_of_flag_triangulations_found += 1
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                            push!(solution_simplices, simplices)
                        end
                        if !config.find_all
                            should_terminate = true 
                        end
                    end
                else # When config.flag_triangulation == false 
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all
                        should_terminate = true
                    end
                end
            else 
                if is_regular(simplices)
                    number_of_regular_triangulations_found += 1
                    if config.flag_triangulation # Looking for quadratic triangulations
                        if is_flag_triangulation(simplices)
                            number_of_quadratic_triangulations_found += 1
                            if isempty(first_solution_simplices)
                                first_solution_simplices = simplices # Maybe this should be "first_quadratic_solution_simplices"
                            end 

                            number_of_quadratic_triangulations_found += 1
                            if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                                push!(solution_simplices, simplices)
                            end
                            if !config.find_all 
                                should_terminate = true 
                            end
                        end

                    else # Only looking for regular unimodular triangulations
                        if isempty(first_solution_simplices)
                                first_solution_simplices = simplices
                        end 
                        if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                                push!(solution_simplices, simplices)
                        end
                        if !config.find_all 
                            should_terminate = true 
                        end
                    end
                end 
            end 
            #= Old logic without the flag condition
            if !config.regular
                if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                    push!(solution_simplices, simplices)
                end
                if !config.find_all; should_terminate = true; end
            else
                if is_regular(simplices)
                    if isempty(first_regular_solution_simplices)
                        first_regular_solution_simplices = simplices
                    end
                    number_of_regular_triangulations_found += 1
                    if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                        push!(solution_simplices, simplices)
                    end
                    if !config.find_all; should_terminate = true; end
                end
            end=#

            if should_terminate
                break
            else
                CadicalWrapper.add_clause(solver, [-solution_vector...]) # Block the clause corresponding to our found solution
            end
        else # UNSAT or Unknown
            if config.flag_triangulation #TODO this is a temporary fix to aviod the return logic
                error("Found an UNSAT instance when searching for flag triangulations. Throwing this error is a temporary fix for now.. ")
            end
            break
        end
    end
    CadicalWrapper.release(solver)
    println("returning solver result?")
    return solution_simplices, first_solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found, number_of_flag_triangulations_found, number_of_quadratic_triangulations_found
end

function solve_cadical_parallel(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool)
    num_threads = Threads.nthreads()
    
    # 1. find generic point and central simplices
    generic_point = find_generic_point(P, internal_faces(P, size(P, 2)), Val(size(P, 2)))
    central_indices_map = compute_central_indices(P, S_indices, generic_point)
    
    # 2. split central simplices into groups for each thread (round-robin)
    central_groups = [Int[] for _ in 1:num_threads]
    for (i, idx) in enumerate(central_indices_map)
        push!(central_groups[mod1(i, num_threads)], idx)
    end

    # 3. prepare thread-local storage for results
    solution_simplices_threads = [Vector{Vector{Matrix{Int}}}() for _ in 1:num_threads]
    first_solution_threads = [Vector{Matrix{Int}}() for _ in 1:num_threads]
    first_regular_threads = [Vector{Matrix{Int}}() for _ in 1:num_threads]
    num_triangulations_threads = zeros(Int, num_threads)
    num_regular_threads = zeros(Int, num_threads)

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
        sol_simp, first_sol, first_reg, num_found, num_reg = solve_cadical_standard(
            local_cnf, P, S_indices, config, false
        )
        
        # Store the returned values into the isolated thread arrays
        solution_simplices_threads[tid] = sol_simp
        first_solution_threads[tid] = first_sol
        first_regular_threads[tid] = first_reg
        num_triangulations_threads[tid] = num_found
        num_regular_threads[tid] = num_reg
    end

    # 5. aggregate the results from all threads
    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    
    number_of_triangulations_found = sum(num_triangulations_threads)
    number_of_regular_triangulations_found = sum(num_regular_threads)

    for tid in 1:num_threads
        append!(solution_simplices, solution_simplices_threads[tid])
        
        if isempty(first_solution_simplices) && !isempty(first_solution_threads[tid])
            first_solution_simplices = first_solution_threads[tid]
        end
        
        if isempty(first_regular_solution_simplices) && !isempty(first_regular_threads[tid])
            first_regular_solution_simplices = first_regular_threads[tid]
        end
    end

    # If only the first triangulation is requested, truncate the combined list
    if config.return_triangulations == "first" && length(solution_simplices) > 1
        resize!(solution_simplices, 1)
    end

    # Print a final summary if requested
    if show_running_updates
        print("Parallel search finished: $number_of_triangulations_found triangulations found ($number_of_regular_triangulations_found regular).")
    end

    return solution_simplices, first_solution_simplices, first_regular_solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found
end

end
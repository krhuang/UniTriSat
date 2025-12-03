module UniTriSat

export triangulate

# Include the Cadical wrapper
include("CadicalWrapper.jl")
using .CadicalWrapper

using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf
using Base.Threads
using TOML
using Random
using CDDLib

# Constants for the new producer-consumer logic
const INTERSECTION_GENERATION_CHUNK_SIZE = 5000 # Number of simplices each generator thread processes at once
const SOLVER_UPDATE_THRESHOLD = 50000  # Number of new clauses to buffer before updating the solver

# mutable flag in module scope
const Normaliz_available = Ref(true)

# Try to import Normaliz.
# If it's not available it gives a small warning and modifies the flag
try
    @eval using Normaliz  # top-level import
    include("Normaliz_backend.jl")
    using .Normaliz_backend
catch e
    Normaliz_available[] = false
end

# mutable flag in module scope
const CUDA_PACKAGES_LOADED = Ref(false)
# try to include Cuda, if its not available and the user wants to use the GPU, then a warning will be printed and we fall back to CPU backend
try
    using CUDA, StaticArrays, CUDA.Adapt
    CUDA_PACKAGES_LOADED[] = true
catch
end
for d in 3:6
    if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_$(d)d.jl")
        include("Intersection_backends/gpu_intersection_$(d)d.jl")
    end
end

# include the rest of the modules
include("Intersection_backends/cpu_intersection.jl")
include("helpers.jl")
using .Helpers
include("basic_computations.jl")
using .BasicComputations
include("plot.jl")
using .Plot
include("subdivision_regularity.jl")
using .SubdivisionRegularity

# Utility: remove ANSI SGR sequences (colors/formatting) from a string.
strip_ansi(s::AbstractString) = replace(s, r"\x1b\[[0-9;]*m" => "")

# a struct to keep track of the timings of the separate operations
struct StepStat
    name::String
    duration_s::Float64
    alloc_bytes::Int64
end

# a struct to aggregate statistics on the fly, avoiding the storage of every single step result
mutable struct StatAggregator
    total_time::Float64
    max_time::Float64
    total_alloc::Int64
    max_alloc::Int64
    count::Int
end
# Initializer for the aggregator
StatAggregator() = StatAggregator(0.0, 0.0, 0, 0, 0)

mutable struct Config
    terminal_output::String
    unimodular::Bool
    intersection_backend::String
    regular::Bool
    find_all::Bool
    validate::Bool
    plot::Bool
    use_normaliz::Bool
    return_triangulations::String
    solver::String
end

mutable struct TriangulationResult
    solution_simplices::Vector{Vector{Matrix{Int}}}
    number_of_triangulations_found::Int
    number_of_regular_triangulations_found::Int
    minimal_log::String
    total_time::Float64
    step_stats::Vector{StepStat}
end

mutable struct RunResult
    triangulation_results::Vector{TriangulationResult}
    number_triangulatable::Int
    number_regularly_triangulatable::Int
    total_number_of_triangulations_found::Int
    total_number_of_regular_triangulations_found::Int
    total_time::Float64
end

# Finds a generic point strictly inside the polytope using Rational{BigInt} arithmetic
function find_generic_point(P_rational::Matrix{Rational{BigInt}}, internal_faces_set, dim::Int)
    n_points_total = size(P_rational, 1)
    max_attempts = 1000

    for attempt in 1:max_attempts
        weights = rand(1:10000, n_points_total)
        weight_sum = sum(weights)
        p_vec = vec((P_rational' * weights) .// weight_sum)

        is_generic = true

        for face_indices in internal_faces_set
            idx_list = collect(face_indices)
            if length(idx_list) < dim
                continue
            end
            v1 = P_rational[idx_list[1], :]
            face_vectors = Matrix{Rational{Int}}(undef, dim, length(idx_list)-1)
            for (i, k) in enumerate(idx_list[2:end])
                face_vectors[:, i] = P_rational[k, :] - v1
            end

            r_face = rank(Float64.(face_vectors))
            if r_face < dim - 1
                continue
            end

            aug_matrix = hcat(face_vectors, p_vec - v1)
            r_aug = rank(Float64.(aug_matrix))

            if r_face == r_aug
                is_generic = false
                break
            end
        end

        if is_generic
            return p_vec
        end
    end
    error("Could not find a generic point.")
end

function is_point_in_simplex(P_rational::Matrix{Rational{BigInt}}, s_indices, p::Vector{Rational{BigInt}})
    dim = length(p)
    indices = collect(s_indices)
    verts = P_rational[indices, :]'
    A = vcat(verts, ones(Rational{BigInt}, 1, dim + 1))
    b = vcat(p, one(Rational{BigInt}))
    try
        lambda = A \ b
        return all(x -> x > 0, lambda)
    catch
        return false
    end
end

# the main function processing a single polytope
function process_polytope(  initial_vertices::Matrix{Int},
                            run_idx::Int,
                            total_in_run::Int,
                            config::Config,
                            show_running_updates::Bool,
                            log_stream::Union{IO, Nothing})

    dim = size(initial_vertices, 2)
    step_stats = Vector{StepStat}()
    t_start_total = time_ns()

    # Printing verbose statements
    function log_verbose(msg...; is_display::Bool=false)
        if isnothing(log_stream)
            return
        end
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = is_display ? sprint(show, "text/plain", msg[1]) : join(msg, " ")
        full_msg = "[$timestamp] " * s_msg
        println(log_stream, full_msg)
    end

    log_verbose("Processing $(dim)D Polytope #$run_idx")
    log_verbose("Initial vertices provided:")
    log_verbose(initial_vertices, is_display=true)

    log_verbose("Step 1: Computing all lattice points...")
    if Normaliz_available[] && config.use_normaliz
        timed_result_lp = @timed lattice_points_via_Normaliz(initial_vertices)
    else
        timed_result_lp = @timed lattice_points_via_CDDLib(initial_vertices)
    end
    P = timed_result_lp.value
    push!(step_stats, StepStat("Compute all lattice points", timed_result_lp.time, timed_result_lp.bytes))

    num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points...")
    end

    simplex_search_type = config.unimodular ? "unimodular" : "non-degenerate"
    log_verbose("Step 2: Computing $simplex_search_type $(dim)-simplices...")

    timed_result_simplices = @timed all_simplices(P, unimodular=config.unimodular)
    S_indices = timed_result_simplices.value
    push!(step_stats, StepStat("Compute $simplex_search_type simplices", timed_result_simplices.time, timed_result_simplices.bytes))

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))
    log_verbose("-> Found $num_simplices simplices. Step 2 complete.\n")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices...")
    end

    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> No simplices found", run_idx, total_in_run, num_lattice_points, num_simplices)
        return TriangulationResult([], 0, 0, minimal_log, time()-t_start_total,step_stats)
    end

    log_verbose("Step 3: Computing internal faces...")
    timed_result_faces = @timed internal_faces(P, dim)
    internal_faces_set = timed_result_faces.value
    push!(step_stats, StepStat("Compute internal faces", timed_result_faces.time, timed_result_faces.bytes))
    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")

    log_verbose("Step 4: Computing intersection clauses (New Logic)...")

    timed_result_intersections = @timed let
        local_clauses = Vector{Vector{Int}}()

        # 4a. Find Generic Point
        P_rational = Matrix{Rational{BigInt}}(P)
        generic_point = find_generic_point(P_rational, internal_faces_set, dim)
        log_verbose("   Generic point found.")

        # 4b. Identify Central Simplices & Compute Full Intersections for them
        central_indices_map = Int[]
        for (i, s) in enumerate(S_indices)
            if is_point_in_simplex(P_rational, s, generic_point)
                push!(central_indices_map, i)
            end
        end
        log_verbose("   Found $(length(central_indices_map)) simplices containing the generic point.")

        if !isempty(central_indices_map)
            central_S_indices = S_indices[central_indices_map]
            #central_clauses = CPUIntersection.get_intersecting_pairs_cpu_generic(P, central_S_indices, Val(dim))
            # all simplices containing the generic point intersect with each other
            central_clauses = [[-i, -j] for i in 1:length(central_S_indices) for j in (i+1):length(central_S_indices)]
            for c in central_clauses
                mapped_clause = [x < 0 ? -central_indices_map[abs(x)] : central_indices_map[abs(x)] for x in c]
                push!(local_clauses, mapped_clause)
            end
        end

        # 4c. Hyperplane Separation Logic
        S_idx_map = Dict(Tuple(sort(collect(s))) => i for (i,s) in enumerate(S_indices))
        for face_indices_iter in combinations(1:num_lattice_points, dim)
            face_indices = collect(face_indices_iter)
            face_verts = [P[i, :] for i in face_indices]
            normal = CPUIntersection.compute_face_normal(face_verts, Val(dim))
            if all(iszero, normal); continue; end

            left_simplices = Int[]
            right_simplices = Int[]
            v_ref = P[face_indices[1], :]

            for p_idx in 1:num_lattice_points
                if p_idx in face_indices; continue; end
                candidate_s = copy(face_indices)
                push!(candidate_s, p_idx)
                sort!(candidate_s)
                candidate_tuple = Tuple(candidate_s)

                if haskey(S_idx_map, candidate_tuple)
                    s_global_idx = S_idx_map[candidate_tuple]
                    val = 0
                    p_coords = P[p_idx, :]
                    for k in 1:dim
                         val += normal[k] * (p_coords[k] - v_ref[k])
                    end
                    if val > 0
                        push!(left_simplices, s_global_idx)
                    elseif val < 0
                        push!(right_simplices, s_global_idx)
                    end
                end
            end

            for i in 1:length(left_simplices)
                s1 = left_simplices[i]
                for j in (i+1):length(left_simplices)
                    s2 = left_simplices[j]
                    push!(local_clauses, [-s1, -s2])
                end
            end
            for i in 1:length(right_simplices)
                s1 = right_simplices[i]
                for j in (i+1):length(right_simplices)
                    s2 = right_simplices[j]
                    push!(local_clauses, [-s1, -s2])
                end
            end
        end
        unique(local_clauses)
    end

    intersection_clauses = timed_result_intersections.value
    push!(step_stats, StepStat("Compute intersection clauses", timed_result_intersections.time, timed_result_intersections.bytes))

    append!(cnf, intersection_clauses)
    log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")

    log_verbose("Step 4d: Generating face-covering clauses...")
    face_dim = dim
    timed_result_face_clauses = @timed let n_simplices = num_simplices
        next_simplex_idx = Threads.Atomic{Int}(1)
        tasks = [
            Threads.@spawn begin
                local_clauses = Vector{Vector{Int}}()
                while true
                    i = Threads.atomic_add!(next_simplex_idx, 1)
                    if i > n_simplices; break; end
                    for face_indices in combinations(S_indices[i], face_dim)
                        canonical_face = Tuple(sort(collect(face_indices)))
                        if canonical_face in internal_faces_set
                            coverers = [j for (j, s2) in enumerate(S_indices) if i != j && issubset(canonical_face, s2)]
                            push!(local_clauses, vcat([-i], coverers))
                        end
                    end
                end
                local_clauses
            end
            for _ in 1:nthreads()
        ]
        vcat(fetch.(tasks)...)
    end
    face_clauses = timed_result_face_clauses.value
    append!(cnf, face_clauses)
    push!(step_stats, StepStat("Generate face-covering clauses", timed_result_face_clauses.time, timed_result_face_clauses.bytes))

    log_verbose("Step 5: Handing SAT problem to solver..."); log_verbose("      Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices solving...")
    end

    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0

    # Determine solver
    active_solver = config.solver
    if active_solver == "cadical"
        if !Sys.islinux()
            @warn("CaDiCaL is currently only supported on Linux. Falling back to PicoSAT.")
            active_solver = "picosat"
        end
    end
    log_verbose("      Using solver: $active_solver")

    timed_solve_result = @timed begin
        if active_solver == "picosat"
            # --- Picosat Logic ---
            solver_func = PicoSAT
            for solution in solver_func.itersolve(cnf)
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
        else
            # --- CaDiCaL Incremental Logic (Producer-Consumer) ---
            solver = CadicalWrapper.Solver()

            # Add initial clauses
            for clause in cnf
                CadicalWrapper.add_clause(solver, clause)
            end

            log_verbose("      (Async) Calculating intersection pairs incrementally and feeding to solver...")

            # 1. Prepare Optimized Simplex Data
            cpu_simplices = CPUIntersection.prepare_simplices_cpu(P, S_indices, Val(dim))

            # 2. Shared State for Intersection Generators
            # Large buffer channel
            clause_channel = Channel{Vector{Int}}(100000)
            
            # Atomic counter for next simplex to check
            next_simplex_idx = Threads.Atomic{Int}(1)
            
            # Flag to indicate if generators are finished
            generation_complete = Threads.Atomic{Bool}(false)

            # 3. Launch Generator Threads
            # We use max(1, nthreads() - 1) to leave the main thread free for coordination
            num_workers = max(1, nthreads() - 1)
            generator_tasks = []
            
            for _ in 1:num_workers
                t = Threads.@spawn begin
                    while true
                        # Grab a chunk of work
                        # The 'i' here represents the index of the first simplex
                        # We will check i against all j > i
                        i = Threads.atomic_add!(next_simplex_idx, 1)
                        if i >= num_simplices
                            break
                        end
                        
                        # Generate conflicts for this simplex against all subsequent ones
                        # Range is (i+1) to num_simplices
                        conflicts = CPUIntersection.check_intersections_range_cpu(cpu_simplices, i, i+1, num_simplices)
                        
                        for c in conflicts
                            put!(clause_channel, c)
                        end
                    end
                end
                push!(generator_tasks, t)
            end

            # Monitor task to close channel when all generators are done
            Threads.@spawn begin
                for t in generator_tasks
                    wait(t)
                end
                generation_complete[] = true
                # Do NOT close the channel immediately, the coordinator needs to drain it.
                # But we can rely on generation_complete + isready checks.
            end

            # 4. Start Solver Async
            task = CadicalWrapper.solve_async(solver)
            number_of_clauses_added = 0
            new_clauses_buffer = Vector{Vector{Int}}()

            # 5. Coordinator Loop
            while true
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
                            # --- Standard Logic ---
                            if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                                push!(solution_simplices, simplices)
                            end
                            if !config.find_all
                                should_terminate = true
                            end
                        else
                            # --- Regular Logic ---
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
                                # It's valid but not regular -> we treat this as a "failure" of the regularity constraint
                                # We don't terminate. We block and continue.
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
                        # If generators are running, but the solver proved UNSAT with a SUBSET of constraints,
                        # adding more constraints (intersections) will not make it SAT.
                        # So UNSAT is final regardless of generator state.
                        break
                    else
                        # Unknown/Interrupted - Just resume
                        task = CadicalWrapper.solve_async(solver)
                    end
                end

                # --- B. Process Incoming Clauses ---
                # Drain channel into local buffer
                while isready(clause_channel)
                    try
                        push!(new_clauses_buffer, take!(clause_channel))
                    catch
                        break
                    end
                end

                # --- C. Update Solver if Threshold Reached ---
                should_update = length(new_clauses_buffer) >= SOLVER_UPDATE_THRESHOLD || (generation_complete[] && !isempty(new_clauses_buffer))
                
                if should_update && !istaskdone(task)
                    number_of_clauses_added += length(new_clauses_buffer)
                    s = "   Number of clauses added: $number_of_clauses_added"
                    print("$s"*"\b"^length(s))
                    CadicalWrapper.interrupt(solver)
                    # We must wait for the task to finish (it should return status 0 or similar)
                    try
                        wait(task)
                    catch
                    end
                    
                    # Add buffered clauses
                    for c in new_clauses_buffer
                        CadicalWrapper.add_clause(solver, c)
                    end
                    empty!(new_clauses_buffer)
                    
                    # Resume solver
                    task = CadicalWrapper.solve_async(solver)
                end
                
                # --- D. Termination Check ---
                # If generators are done, buffer is empty, and solver is running (and not finished in A), just wait.
                if generation_complete[] && isempty(new_clauses_buffer) && !istaskdone(task)
                    sleep(0.01) # Yield
                else
                    sleep(0.001) # Yield slightly to let generators work
                end
            end
            
            CadicalWrapper.release(solver)
        end
    end


    
    push!(step_stats, StepStat("Solve SAT problem", timed_solve_result.time, timed_solve_result.bytes))
    log_verbose("-> SAT solver finished. Step 5 complete.")

    log_verbose("\n$(number_of_triangulations_found) valid triangulation(s) found.")
    if config.regular
        log_verbose("\n$(number_of_regular_triangulations_found) valid regular triangulation(s) found.")
    end

    if !isempty(first_solution_simplices) && number_of_regular_triangulations_found > 0
        log_verbose("\nDisplaying first valid triangulation:")
        for s in first_solution_simplices
            log_verbose(s, is_display=true)
        end
    end
    if !isempty(first_regular_solution_simplices)
        log_verbose("\nDisplaying first valid regular triangulation:")
        for s in first_regular_solution_simplices
            log_verbose(s, is_display=true)
        end
    end

    if config.plot
        log_verbose("\nStep 6: Plotting result..")
        if config.regular
            if isempty(first_regular_solution_simplices)
                @error("Cannot plot, no regular triangulation found")
             else
                plot(initial_vertices, dim, first_regular_solution_simplices)
            end
        else
            if isempty(first_solution_simplices)
                @error("Cannot plot, no triangulation found")
            else
                 plot(initial_vertices, dim, first_solution_simplices)
            end
        end
        log_verbose("-> Plotting complete. Step 6 complete.")
    end

    total_time = (time_ns() - t_start_total) / 1e9

    summary_buf = IOBuffer()
    println(summary_buf, "\n--- Timing & Memory Summary for Polytope #$run_idx ---")
    println(summary_buf, @sprintf("%-45s | %-12s | %-12s", "Step", "Duration", "Allocations"))
    println(summary_buf, "-"^73)
    for stat in step_stats
        println(summary_buf, @sprintf("%-45s | %-12s | %-12s", stat.name, @sprintf("%.4f s", stat.duration_s), format_bytes(stat.alloc_bytes)))
    end
    println(summary_buf, "-"^73)
    println(summary_buf, @sprintf("%-45s | %-12s | %-12s", "Total execution time", @sprintf("%.4f s", total_time), ""))
    peak_ram_bytes = Sys.maxrss()
    println(summary_buf, @sprintf("%-45s: %.2f MiB", "Peak memory usage (Max RSS)", peak_ram_bytes / 1024^2))
    log_verbose(String(take!(summary_buf)))

    result_str = ""
    if number_of_regular_triangulations_found > 0
        result_str = @sprintf("\u001b[32mfound %d regular solution(s)\u001b[0m in %.2f s", number_of_regular_triangulations_found, total_time)
    elseif number_of_triangulations_found > 0
        result_str = @sprintf("\u001b[32mfound %d solution(s)\u001b[0m in %.2f s", number_of_triangulations_found, total_time)
    else
        result_str = @sprintf("\u001b[31mno solution exists\u001b[0m, searched for %.2f s", total_time)
    end
    minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> %s", run_idx, total_in_run, num_lattice_points, num_simplices, result_str)

    empty!(cnf)
    cnf = Vector{Vector{Int}}()

    return TriangulationResult(solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found, minimal_log, total_time, step_stats)
end

function run_processing(polytopes::Vector{Matrix{Int}}, config::Config, log_stream)
    number_of_polytopes = length(polytopes)
    components_str = lowercase(replace(config.terminal_output, " " => ""))

    show_initial = occursin("initial", components_str) || occursin("all", components_str)
    show_running = occursin("running", components_str) || occursin("all", components_str)
    show_table = occursin("table", components_str) || occursin("all", components_str)
    show_final = occursin("final", components_str) || occursin("all", components_str)

    if show_initial
        term_summary_buf = IOBuffer()
        println(term_summary_buf, "Run started at:                      $(Dates.format(now(), "HH:MM:SS"))")
        println(term_summary_buf, "Number of threads:                   $(nthreads())")
        println(term_summary_buf, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
        println(term_summary_buf, "Solver:                              $(config.solver)")
        println(term_summary_buf, "Intersection backend selected:       $(config.intersection_backend)")
        println(term_summary_buf, "Validation enabled:                  $(config.validate)")
        println(term_summary_buf, "Number of polytopes found:           $(length(polytopes))")
        println(term_summary_buf, "Restricting to unimodular simplices: $(config.unimodular)")
        println(term_summary_buf, "Looking for regular triangulations:  $(config.regular)")
        println(term_summary_buf, "Using Normaliz:                      $(config.use_normaliz)")
        println(term_summary_buf, "")
         print(stdout, String(take!(term_summary_buf)))
    end

    if !isnothing(log_stream)
        log_summary_buf = IOBuffer()
        println(log_summary_buf, "Number of threads:                      $(nthreads())")
        println(log_summary_buf, "Solve mode:                             $(config.find_all ? "Find All" : "Find First")")
        println(log_summary_buf, "Solver:                                 $(config.solver)")
        println(log_summary_buf, "Intersection backend selected:          $(config.intersection_backend)")
        println(log_summary_buf, "Validation enabled:                     $(config.validate)")
         println(log_summary_buf, "Number of polytopes found:              $(length(polytopes))")
        println(log_summary_buf, "Restricting to unimodular simplices:    $(config.unimodular)")
        println(log_summary_buf, "Looking for regular triangulations:     $(config.regular)")
        println(log_summary_buf, "Using Normaliz:                         $(config.use_normaliz)")
        println(log_summary_buf, "")
        print(log_stream, String(take!(log_summary_buf)))
        flush(log_stream)
    end

    if config.use_normaliz && !Normaliz_available[]
        @warn(
            """
            \n====================================== WARNING ======================================
            Normaliz not available; using CDDLib lattice point enumeration instead.
            This is slower, but not the bottleneck, so it should be OK.
            You can find Normaliz.jl at https://github.com/Normaliz/Normaliz.jl
            You may have to downgrade your Julia version for Normaliz to work.
            There is also the lattice_points_via_Oscar function available in basic_computation.jl
            =====================================================================================\n
            """)
    end

    t_start_run = time()
    triangulatable = 0
    regularly_triangulatable = 0

    total_number_of_triangulations_found = 0
    total_number_of_regular_triangulations_found = 0

    is_first_single_line_update = true

    global_step_stats = Dict{String, StatAggregator}()
    step_order = String[]
    all_results = Vector{TriangulationResult}()

    for (i, P) in enumerate(polytopes)

         r = process_polytope(P, i, length(polytopes), config, show_running, log_stream)
        number_of_triangulations_found = r.number_of_triangulations_found
        number_of_regular_triangulations_found = r.number_of_regular_triangulations_found
        minimal_log = r.minimal_log
        step_stats = r.step_stats

        push!(all_results, r)

        if !isnothing(log_stream)
             flush(log_stream)
        end

         if number_of_triangulations_found > 0
            triangulatable += 1
        end
        if number_of_regular_triangulations_found > 0
            regularly_triangulatable += 1
        end
        total_number_of_triangulations_found += number_of_triangulations_found
        total_number_of_regular_triangulations_found += number_of_regular_triangulations_found

        for stat in step_stats
             if !haskey(global_step_stats, stat.name)
                global_step_stats[stat.name] = StatAggregator()
                push!(step_order, stat.name)
            end
            agg = global_step_stats[stat.name]
            agg.total_time += stat.duration_s
            agg.max_time = max(agg.max_time, stat.duration_s)
             agg.total_alloc += stat.alloc_bytes
            agg.max_alloc = max(agg.max_alloc, stat.alloc_bytes)
            agg.count += 1
        end

        if show_running
            if !is_first_single_line_update
                print(stdout, "\u001b[4A")
                if config.regular; print(stdout, "\u001b[1A"); end
            end
            is_first_single_line_update = false
            elapsed_time = time() - t_start_run
            eta_str = ""
            avg_time = (time() - t_start_run) / i
            remaining = number_of_polytopes - i
             eta_seconds = avg_time * remaining
            eta_str = format_duration(eta_seconds)

            @printf(stdout, "\r%-40s %s\u001b[K\n", "Elapsed Time:", format_duration(elapsed_time))
            @printf(stdout, "\r%-40s %s\u001b[K\n", "Estimated Time Left:", eta_str)
            if config.regular; @printf(stdout, "\r%-40s \u001b[32m%d\u001b[0m\u001b[K\n", "Regularly Triangulatable:", regularly_triangulatable); end
            @printf(stdout, "\r%-40s \u001b[32m%d\u001b[0m\u001b[K\n", "Triangulatable:", triangulatable)
            @printf(stdout, "\r%-40s \u001b[31m%d\u001b[0m\u001b[K\n", "Non-Triangulatable:", i - triangulatable)
            print(stdout, "\r" * minimal_log * "\u001b[K")
            flush(stdout)
        end
        step_stats = nothing
    end

    if show_running
         print(stdout, "\u001b[5A")
        if config.regular; print(stdout, "\u001b[1A"); end
        print(stdout, "\u001b[0J")
    end

    total_time_run = time() - t_start_run

    avg_solutions_str = ""
    if config.find_all
        num_sol = config.regular ? total_number_of_regular_triangulations_found : total_number_of_triangulations_found
        avg_solutions_str = @sprintf("Average Solutions/Polytope:      %.2f\n", num_sol / number_of_polytopes)
    end
    stats_table_str = ""
    if show_table
        stats_table_buf = IOBuffer()
        if !isempty(global_step_stats)
            println()
            println(stats_table_buf, @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s | %-12s",
                                      "Step Name", "Total Time", "Avg Time", "Max Time", "Avg Memory", "Max Memory"))
            println(stats_table_buf, "-"^108)

            for step_name in step_order
                if !haskey(global_step_stats, step_name); continue; end
                stat = global_step_stats[step_name]

                total_time = stat.total_time
                max_time = stat.max_time
                avg_time = total_time / stat.count
                avg_mem = stat.total_alloc / stat.count

                 println(stats_table_buf, @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s | %-12s",
                                                step_name,
                                                 format_duration(total_time),
                                                @sprintf("%.3f s", avg_time),
                                                 @sprintf("%.3f s", max_time),
                                                format_bytes(avg_mem),
                                                 format_bytes(stat.max_alloc)))
            end
        end
        stats_table_str = String(take!(stats_table_buf))
    end

    reg_str = config.regular ? "\nRegularly Triangulatable:      \u001b[32m$regularly_triangulatable\u001b[0m" : ""
    summary_core_str = """

    Run finished: $(Dates.format(now(), "HH:MM:SS"))

    ----------------------------------------
    Run Summary
    ----------------------------------------
    Total Polytopes Processed:     $(length(polytopes))$(reg_str)
    Triangulatable:                \u001b[32m$triangulatable\u001b[0m
    Non-Triangulatable:            \u001b[31m$(number_of_polytopes - triangulatable)\u001b[0m
    $(avg_solutions_str)Total Run Time:          $(format_duration(total_time_run))
    ----------------------------------------
    """
    if show_final
        print(stdout, summary_core_str)
    end
    if show_table
        print(stdout, stats_table_str)
        println(stdout)
    end

    if !isnothing(log_stream)
        print(log_stream, strip_ansi(summary_core_str))
        print(log_stream, strip_ansi(stats_table_str))
        flush(log_stream)
    end

    return RunResult(all_results, triangulatable, regularly_triangulatable, total_number_of_triangulations_found, total_number_of_regular_triangulations_found, time()-t_start_run)
end

function setup_run( polytopes::Vector{Matrix{Int}},
                    intersection_backend::String="cpu",
                    unimodular::Bool=true, regular::Bool=false,
                    find_all::Bool=false, log_file::String="",
                    terminal_output::String="",
                   validate::Bool=false,
                    plot::Bool=false,
                    use_normaliz::Bool=false,
                    return_triangulations::String="first",
                    solver::String="picosat")

    config = Config(terminal_output, unimodular, intersection_backend, regular, find_all, validate, plot, use_normaliz, return_triangulations, solver)
    log_stream = nothing
    results = Vector{Vector{Vector{Matrix{Int}}}}()
    try
        if !isempty(log_file)
            try
                log_stream = open(log_file, "a")
                println(log_stream, "\n\n" * "#"^80, "\n# New Run Started at $(now())\n" * "#"^80)
            catch e
                 @error("Error opening log file: $e")
                log_stream = nothing
            end
        end

        if isempty(polytopes)
            @warn("No polytopes provided to setup_run.")
            return results
        end
    
     results = run_processing(polytopes, config, log_stream)
    finally
        !isnothing(log_stream) && close(log_stream)
    end
    return results
end

function triangulate(   vmatrix::Matrix{Int};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
     return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver)
end

function triangulate(   vmatrices::Vector{Matrix{Int}};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver)
end

function triangulate(   polytope::Polyhedron;
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end

    vmatrix = _convert_polyhedron_to_vmatrix(polytope)
    if isempty(vmatrix)
        @error("Could not process a single polytope")
        return nothing
    end
    return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver)
end

function triangulate(   polytopes::Vector{Polyhedron};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
    vmatrices = Matrix{Int}[]
    for p in polytopes
        vmatrix = _convert_polyhedron_to_vmatrix(p)
        if !isempty(vmatrix)
            push!(vmatrices, vmatrix)
        else
            @warn("Scipping a polytopes, because it could not be read properly-.")
        end
    end

    if isempty(vmatrices)
        @error("Could not porcess a single polytope.")
         return Vector{Vector{Matrix{Int}}}[]
    end
    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver)
end

function triangulate(   path_to_polytopes::String;
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
    local polytopes
     try
        polytopes = read_polytopes_from_file(path_to_polytopes)
        if isempty(polytopes); @error("Error: No polytopes loaded from '$path_to_polytopes'."); return Vector{Vector{Matrix{Int}}}[]; end
        catch e
        @error("Error loading polytopes from '$path_to_polytopes': '$e'")
        return Vector{Vector{Matrix{Int}}}[]
    end
    return setup_run(polytopes, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver)
end

end

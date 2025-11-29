module UniTriSat

export triangulate

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
# Matches ESC '[' followed by digits and semicolons and ending in 'm', e.g. "\x1b[1;31m".
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

# contains the run details
# terminal_output (default "") is a subset of "initial, running, table, final".
# Depending on which are present, the following is printed to the terminal:
#       initial: print an initial run summary
#       running: after a polytope is done, a running summary is being updated showing intermediate results
#       table: after the run is done, a table breaking up the time and memory used in each major computation step
#       final: print a final summary showing the number of solutions found and the total run time
# unimodular (default true): if true, only unimodular 
# simplices are used for the triangulations
# intersection backend (default "cpu"): select between the backends "cpu" and "gpu"
# regular (default false): if true, we search for regular triangulations
# find_all (default false): if true, then we enumerate all triangulations
# validate (default false): NOT YET IMPLEMENTED
# plot (default false): if true, the first triangulations found is being plotted.
# If the dimension is not 3, then we plot the 3-faces
# use_normaliz (default false): if true, we use the much faster, but unstable normaliz backend to find all lattice points in a given polytope
# return_triangulations (default "first"): Select between "", "first" and "all".
# The returned vector will be empty, contain the first set of
# simplices of a solution, or contain all solutions, respectively.
# On long runs the set of solutions can clog up the memory if set to "all" (with find_all set to true)
#and on really long runs even "first" is not recommended.
# Note that the first found solution can also be recovered from the log file.
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

# the main function processing a single polytope, here all 
# the computations happen
# Modified to return a Tuple instead of ProcessResult to reduce memory overhead
function process_polytope(  initial_vertices::Matrix{Int}, 
                            run_idx::Int,
                            total_in_run::Int,
                            config::Config, 
                            show_running_updates::Bool,
                            log_stream::Union{IO, Nothing})

    dim = size(initial_vertices, 2)
    step_stats = Vector{StepStat}()
    t_start_total = time_ns()
    validation_status = :not_run

    # Printing verbose statements
    function log_verbose(msg...; is_display::Bool=false)
        if isnothing(log_stream)
            return
        end
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = if is_display
            sprint(show, "text/plain", msg[1])
        else
            join(msg, " ")
        end
        full_msg = "[$timestamp] " * s_msg
        println(log_stream, full_msg)
    end

    log_verbose("Processing $(dim)D Polytope #$run_idx")
    log_verbose("Initial vertices provided:")
    log_verbose(initial_vertices, is_display=true)

    log_verbose("Step 1: Computing all lattice points...")
    if Normaliz_available[] && config.use_normaliz # global flag for if the Normaliz package has been imported
        timed_result_lp = @timed lattice_points_via_Normaliz(initial_vertices) # Find the lattice points.
        # Source in Normaliz_backend.jl
    else
        timed_result_lp = @timed lattice_points_via_CDDLib(initial_vertices)
    end
    P = timed_result_lp.value # P is now the set of all lattice points in the polytope
    push!(step_stats, StepStat("Compute all lattice points", timed_result_lp.time, timed_result_lp.bytes))

    num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points...")
    end

    simplex_search_type = config.unimodular ? "unimodular" : "non-degenerate"
    log_verbose("Step 2: Computing $simplex_search_type $(dim)-simplices...")

    timed_result_simplices = @timed all_simplices(P, unimodular=config.unimodular) # find all (unimodular) simplices spanned by P
    S_indices = timed_result_simplices.value
    push!(step_stats, StepStat("Compute $simplex_search_type simplices", timed_result_simplices.time, timed_result_simplices.bytes))

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices)) # set up the cnf formula.
    # This first clause makes sure that at least one simplex must be chosen for the triangulation
    log_verbose("-> Found $num_simplices simplices. Step 2 complete.\n")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices...")
    end

    if isempty(S_indices) # handle the case that there are no simplices
        total_time = (time_ns() - t_start_total) / 1e9
        minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> No simplices found", run_idx, total_in_run, num_lattice_points, num_simplices)
        return TriangulationResult([], 0, 0, minimal_log, time()-t_start_total,step_stats)
    end

    log_verbose("Step 3: Computing internal faces...")
    # the internal faces are d-1 dimensional simplices which are not contained in a facet
    # each of these must be a facet of exactly two or exactly zero simplices used in a triangulation
    # we do not check these for unimodularity, as the SAT solver does not mind a few extra clauses and the checking would take real time

    timed_result_faces = @timed internal_faces(P, dim)
    internal_faces_set = timed_result_faces.value
    push!(step_stats, StepStat("Compute internal faces", timed_result_faces.time, timed_result_faces.bytes))
    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")

    log_verbose("Step 4: Computing intersecting pairs...")
    # the main part of the computation is finding all pairs of simplices which intersect with volume

    timed_result_intersections = @timed let n_simplices = num_simplices
        intersect_func = nothing
        use_gpu = false

        # load the right GPU backend if required, or fall back to CPU
        if config.intersection_backend == "gpu"
            if dim == 3 && isdefined(@__MODULE__, :GPUIntersection3D)
                log_verbose("     Using 3D GPU backend...")
                intersect_func = () -> GPUIntersection3D.get_intersecting_pairs_gpu(P, S_indices)
                use_gpu = true
            elseif dim == 4 && isdefined(@__MODULE__, :GPUIntersection4D)
                log_verbose("     Using 4D GPU backend...")
                intersect_func = () -> GPUIntersection4D.get_intersecting_pairs_gpu_4d(P, S_indices)
                use_gpu = true
            elseif dim == 5 && isdefined(@__MODULE__, :GPUIntersection5D)
                log_verbose("     Using 5D GPU backend...")
                intersect_func = () -> GPUIntersection5D.get_intersecting_pairs_gpu_5d(P, S_indices)
                use_gpu = true
            elseif dim == 6 && isdefined(@__MODULE__, :GPUIntersection6D)
                log_verbose("     Using 6D GPU backend...")
                intersect_func = () -> GPUIntersection6D.get_intersecting_pairs_gpu_6d(P, S_indices)
                use_gpu = true
            end
        end
        if use_gpu && !isnothing(intersect_func)
            intersect_func() # Execute the selected GPU function
        else
            if config.intersection_backend == "gpu"
                log_verbose("     WARNING: GPU backend for $(dim)D not available. Falling back to CPU.")
            end
            log_verbose("     Using CPU backend.")
            CPUIntersection.get_intersecting_pairs_cpu_generic(P, S_indices, Val(dim))
        end
    end

    intersection_clauses = timed_result_intersections.value
    push!(step_stats, StepStat("Compute intersecting pairs", timed_result_intersections.time, timed_result_intersections.bytes))


    append!(cnf, intersection_clauses)
    #add the clauses to the formula. If s1 and s2 intersect, then (not 
    # s1) or (not s2) is added, ensuring that not both can be in a triangulation at the same time
    log_verbose("-> Found $(length(intersection_clauses)) intersecting pairs (after filtering). Step 4c complete.\n")

    log_verbose("Step 4d: Generating face-covering clauses...")
    # we already computed the set of internal faces, we still need to compute the clauses to add them to the formula
    # If simplex s has internal face f as a facet, and s1,...,sk is the set of simplices other than s which have f as a facet, then we add the clause
    # (not s) or s1 or s2 or s3 or ...
    # together with the intersection clauses, this implies that exactly zero or two simplices contain f
    face_dim = dim
    timed_result_face_clauses = @timed let n_simplices = num_simplices
        next_simplex_idx = Threads.Atomic{Int}(1)
        tasks = [
            Threads.@spawn begin
                local_clauses = Vector{Vector{Int}}()
                while true
                    i = Threads.atomic_add!(next_simplex_idx, 1)
                    if i > n_simplices
                        break
                    end
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
    
    log_verbose("-> Found $(length(face_clauses)) face-covering clauses. Step 4d complete.\n")

    log_verbose("Step 5: Handing SAT problem to solver...");
    log_verbose("      Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")
    if show_running_updates
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices solving...")
    end

    # for line in cnf
    #     println(line)
    # end
    # exit()

    solution_simplices = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()
    first_regular_solution_simplices = Vector{Matrix{Int}}()
    number_of_triangulations_found = 0
    number_of_regular_triangulations_found = 0
    solver_func = PicoSAT # we only support PicoSAT atm, but other solvers could be easily used by replacing this solver function
    # (and possible changing how the solver api is called)

    timed_solve_result = @timed for solution in solver_func.itersolve(cnf)
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
            if !config.find_all; break; end #we found a solution, we do not want a regular one and we dont want all of them : We can stop here
        end
        reg = is_regular(simplices)
        if config.regular
            if reg
                if isempty(first_regular_solution_simplices)
                    first_regular_solution_simplices = simplices
                end
                number_of_regular_triangulations_found += 1
                if config.return_triangulations == "all" || (config.return_triangulations == "first" && isempty(solution_simplices))
                    push!(solution_simplices, simplices)
                end
                if !config.find_all; break; end #we found a regular solution and we dont want all of them : We can stop here
            elseif show_running_updates
                s = " ($number_of_triangulations_found non-regular triangulations found)"
                print(s*"\b"^(length(s)))
            end
        end
    end

    if number_of_triangulations_found > 0 && number_of_regular_triangulations_found == 0
        # This logic seems to be a debug break from the original code, kept as requested.
        # println(initial_vertices)
        # exit()
    end

    num_solutions = config.regular ? number_of_regular_triangulations_found : number_of_triangulations_found

    push!(step_stats, StepStat("Solve SAT problem", timed_solve_result.time, timed_solve_result.bytes))
    log_verbose("-> SAT solver finished. Step 5 complete.")

    # validation is not yet implemented. We plan to have very robust and trusted code (using exact Rational{BigInt}) test everything again
#     if config.validate && num_solutions > 0
#         log_verbose("\nStep 6: Validating solution (not yet implemented)...")
#         timed_validation = @timed begin
#             validation_status = :passed
#             #TODO implement validation or remove validation
#         end
#
#         push!(step_stats, StepStat("Validation", timed_validation.time, timed_validation.bytes))
#         if validation_status == :passed
#             log_verbose("  VALIDATION SUCCESSFUL: No intersections found among solution simplices.")
#         else
#             @error("Valitdation failed! Initial vertices where: '$initial_vertices'")
#         end
#         log_verbose("-> Validation complete. Step 6 complete.")
#     end

    log_verbose("\n$(number_of_triangulations_found) valid triangulation(s) found.")
    if config.regular
        log_verbose("\n$(number_of_regular_triangulations_found) valid regular triangulation(s) found.")
    end

    if !isempty(first_solution_simplices) && number_of_regular_triangulations_found > 0
        log_verbose("\nDisplaying first valid triangulation:");
        for s in first_solution_simplices
            log_verbose(s, is_display=true)
        end
    end
    if !isempty(first_regular_solution_simplices)
        log_verbose("\nDisplaying first valid regular triangulation:");
        for s in first_regular_solution_simplices
            log_verbose(s, is_display=true)
        end
    end

    # plotting uses the python script plot_triangulation.py found under src/
    # we plot the intersection of the triangulation from the first solution with every facet of the polytope if it is not 3d
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
    cnf = Vector{Vector{Int}}() # Reassign to empty to allow GC to eat the old one immediately

    # Return tuple: (solutions, step_stats, num_found, num_regular, min_log, total_time)
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
        println(log_summary_buf, "Number of threads:                    $(nthreads())")
        println(log_summary_buf, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
        println(log_summary_buf, "Intersection backend selected:       $(config.intersection_backend)")
        println(log_summary_buf, "Validation enabled:                  $(config.validate)")
        println(log_summary_buf, "Number of polytopes found:           $(length(polytopes))")
        println(log_summary_buf, "Restricting to unimodular simplices: $(config.unimodular)")
        println(log_summary_buf, "Looking for regular triangulations:  $(config.regular)")
        println(log_summary_buf, "Using Normaliz:                      $(config.use_normaliz)")
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
    # all_solutions = Vector{Vector{Vector{Matrix{Int}}}}()
    # if config.return_triangulations != ""
    #     sizehint!(all_solutions, number_of_polytopes)
    # end

    global_step_stats = Dict{String, StatAggregator}()
    step_order = String[] # To preserve the order of steps for the final table
    all_results = Vector{TriangulationResult}()

    for (i, P) in enumerate(polytopes)

        r = process_polytope(P, i, length(polytopes), config, show_running, log_stream)
        # solution_simplices = r.solution_simplices
        number_of_triangulations_found = r.number_of_triangulations_found
        number_of_regular_triangulations_found = r.number_of_regular_triangulations_found
        minimal_log = r.minimal_log
        # total_time = r.total_time
        step_stats = r.step_stats

        push!(all_results, r)

        # Write to log if needed
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

        # Online aggregation of statistics
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

        # # Store solutions only if requested
        # if config.return_triangulations != ""
        #     push!(all_solutions, solution_simplices)
        # end

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

        # Explicitly clear temporary variables to help GC
        # solutions = nothing
        step_stats = nothing

#         if i%1000 == 0
#             GC.gc()
#             ccall(:malloc_trim, Cvoid, (Cint,), 0)
#         end
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

            # Iterate over the preserved order of steps
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
    $(avg_solutions_str)Total Run Time:                  $(format_duration(total_time_run))
    ----------------------------------------
    """
    if show_final
        print(stdout, summary_core_str)
    end
    if show_table
        print(stdout, stats_table_str)
        println(stdout)
    end
    # Strip ANSI SGR color/formatting sequences from log output before writing to file.
    # Existing regex used "\\u001b\[\d+m" which misses multi-parameter sequences like "\x1b[1;34m".
    # Use a more general pattern that matches the ESC (hex 1b) followed by '[' and any digits/semicolon params ending in 'm'.
    if !isnothing(log_stream)
        print(log_stream, strip_ansi(summary_core_str))
        print(log_stream, strip_ansi(stats_table_str))
        flush(log_stream)
    end

    return RunResult(all_results, triangulatable, regularly_triangulatable, total_number_of_triangulations_found, total_number_of_regular_triangulations_found, time()-t_start_run)
end

# the entry point of the internal code of the modul. It sets up the config struct, opens the log file etc.
function setup_run( polytopes::Vector{Matrix{Int}}, 
                    intersection_backend::String="cpu", 
                    unimodular::Bool=true, regular::Bool=false, 
                    find_all::Bool=false, log_file::String="", 
                    terminal_output::String="", 
                    validate::Bool=false, 
                    plot::Bool=false, 
                    use_normaliz::Bool=false, 
                    return_triangulations::String="first")

    config = Config(terminal_output, unimodular, intersection_backend, regular, find_all, validate, plot, use_normaliz, return_triangulations)
    log_stream = nothing
    # Initialize with empty vector instead of typed empty array for type stability
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

# Public api function
# it can be called with a matrix containing the vertices, a list of matrices, Polyhedra object(s) or a path to a file from which to read in the polytopes

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
                        return_triangulations::String="first")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
     return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations)
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
                        return_triangulations::String="first")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end
    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations)
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
                        return_triangulations::String="first")

    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end

    vmatrix = _convert_polyhedron_to_vmatrix(polytope)
    if isempty(vmatrix)
        @error("Could not process a single polytope")
        return nothing
    end
    return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations)
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
                        return_triangulations::String="first")

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
    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations)
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
                        return_triangulations::String="first")

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
    return setup_run(polytopes, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations)
end

end

module UniTriSat


export triangulate

using Combinatorics
using LinearAlgebra
using Polyhedra
using Dates
using Printf
using Base.Threads
using Random
using CDDLib
using StaticArrays
using AbstractAlgebra

include("structs.jl")
using .Structs
include("precision.jl")
include("helpers.jl")
using .Helpers
include("basic_computations.jl")
using .BasicComputations
include("plot.jl")
using .Plot
include("solve.jl")
using .Solving

# Utility: remove ANSI SGR sequences (colors/formatting) from a string.
strip_ansi(s::AbstractString) = replace(s, r"\x1b\[[0-9;]*m" => "")

# the main function processing a single polytope
function process_polytope(  initial_vertices::Matrix{Int},
                            run_idx::Int,
                            total_in_run::Int,
                            config::Config,
                            show_running_updates::Bool,
                            log_stream::Union{IO, Nothing})

     # Printing verbose statements
    function log_verbose(msg...; is_display::Bool=false)
        if isnothing(log_stream)
            return
        end
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = is_display ? sprint(show, "text/plain", msg[1]) : join(msg, " ")
        full_msg = "[$timestamp] " * s_msg
        println(log_stream, full_msg)
        flush(log_stream)
    end

    dim = size(initial_vertices, 2)
    step_stats = Vector{StepStat}()
    t_start_total = time_ns()

    log_verbose("Processing $(dim)D Polytope #$run_idx")
    log_verbose("Initial vertices provided:")
    log_verbose(initial_vertices, is_display=true)

    # --- Step 0: Check full-dimensionality (optional) ---
    #           Checks if our polytope is full-dimensional. 
    #           When it's not, we compute its Hermite Normal Form and find a lattice-equivalent polytope 
    #           in a lower ambient dimension.

    dim = size(initial_vertices, 2)
    if config.check_full_dimensionality
        poly = polyhedron(vrep(initial_vertices), CDDLib.Library(:exact))
        intrinsic_dim = Polyhedra.dim(poly)
        if intrinsic_dim < dim
            log_verbose("Polytope is not full-dimensional. Original ambient dim: $dim, Polytope intrinsic dim: $intrinsic_dim")
            log_verbose("Finding an appropriate projection to remove excess ambient dimensions via HNF...")

            # Execute the lattice-preserving projection transformation
            initial_vertices = full_dimensional_lattice_projection(initial_vertices)
            dim = size(initial_vertices, 2)

            # Rebuild the Polyhedron structure in the new lower-dimensional space
            poly = polyhedron(vrep(initial_vertices), CDDLib.Library(:exact))
            dim_poly = Polyhedra.dim(poly)

            # Sanity Check
            if dim_poly != dim
                error("Projection failed to produce a full-dimensional polytope. Target ambient dim: $dim, but got: $dim_poly. \nThis error should never occur; please write us.")
            end

            log_verbose("Successfully projected vertices to full-dimensional lattice space.")
            log_verbose(initial_vertices, is_display=true)
        else
            log_verbose("Polytope is already full-dimensional. Continuing with original vertices.")
        end
    else
        log_verbose("Polytope is already full-dimensional. Continuing with original vertices.")
    end

    # --- Step 1: Lattice Points ---
    log_verbose("Step 1: Computing all lattice points...")
    timed_result_lp = @timed compute_lattice_points(initial_vertices, config)
    P = timed_result_lp.value
    push!(step_stats, StepStat("Compute all lattice points", timed_result_lp.time, timed_result_lp.bytes))

    num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if show_running_updates
        ghost_print("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points...")
    end

    # --- Step 2: Simplices ---
    simplex_search_type = config.unimodular ? "unimodular" : "non-degenerate"
    log_verbose("Step 2: Computing $simplex_search_type $(dim)-simplices...")

    timed_result_simplices = @timed compute_simplices(P, config)
    S_indices = timed_result_simplices.value
    simplex_step_name = config.unimodular ? "Compute unimodular simplices" : "Compute non-degenerate simplices"
    push!(step_stats, StepStat(simplex_step_name, timed_result_simplices.time, timed_result_simplices.bytes))

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))
    log_verbose("-> Found $num_simplices simplices. Step 2 complete.\n")
    if show_running_updates
        ghost_print("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices...")
    end

    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        return TriangulationResult([], 0, 0, 0, 0, total_time, step_stats)
    end

    # --- Step 3: Internal Faces ---
    log_verbose("Step 3: Computing internal faces...")
    timed_result_faces = @timed compute_internal_faces(P, dim)
    internal_faces_set = timed_result_faces.value
    push!(step_stats, StepStat("Compute internal faces", timed_result_faces.time, timed_result_faces.bytes))
    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")
    
    # initialize an intersection_matrix
    n = length(S_indices) 
    intersection_matrix = [falses(n) for _ in 1:n]
    # --- Step 4: Intersections ---
    
    if config.incremental_solving
        log_verbose("Step 4a: Computing intersection clauses...")
        timed_result_intersections = @timed compute_intersections_incremental(P, S_indices, internal_faces_set, dim, num_lattice_points)
        intersection_clauses = timed_result_intersections.value
        if config.flag_SAT # Build a data structure for flag_SAT computations here
            for clause in intersection_clauses 
                a, b = abs(clause[1]), abs(clause[2])
                intersection_matrix[a][b] = true
                intersection_matrix[b][a] = true 
            end
        end
        push!(step_stats, StepStat("Compute intersection clauses", timed_result_intersections.time, timed_result_intersections.bytes))
        append!(cnf, intersection_clauses)
        log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")
    
    elseif config.circuit_intersection_clauses  
        log_verbose("Step 4a: Computing intersecting simplices via circuits")
        timed_result_intersections = @timed compute_intersections_circuits(P, S_indices, dim, config, log_verbose)
        intersection_clauses = timed_result_intersections.value 
        if config.flag_SAT 
            for clause in intersection_clauses 
                a, b = abs(clause[1]), abs(clause[2])
                intersection_matrix[a][b] = true
                intersection_matrix[b][a] = true 
            end
        end
        push!(step_stats, StepStat("Compute intersection clauses", timed_result_intersections.time, timed_result_intersections.bytes))
        append!(cnf, intersection_clauses)
        log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")
    else
        # Non-incremental: Use provided backend logic to find all intersections
        log_verbose("Step 4a: Computing intersecting pairs via candidate separating hyperplanes...")
        timed_result_intersections = @timed compute_intersections_standard(P, S_indices, dim, config, log_verbose)
        intersection_clauses = timed_result_intersections.value
        if config.flag_SAT # Build a data structure for flag_SAT computations here
            for clause in intersection_clauses 
                a, b = abs(clause[1]), abs(clause[2])
                intersection_matrix[a][b] = true
                intersection_matrix[b][a] = true 
            end
        end
        push!(step_stats, StepStat("Compute intersecting pairs", timed_result_intersections.time, timed_result_intersections.bytes))
        append!(cnf, intersection_clauses)
        log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")
    end

    # --- Step 4b: Face-Covering Clauses ---
    log_verbose("Step 4b: Generating face-covering clauses...")
    timed_result_face_clauses = @timed compute_face_clauses(S_indices, internal_faces_set, dim)
    face_clauses = timed_result_face_clauses.value
    append!(cnf, face_clauses)
    push!(step_stats, StepStat("Generate face-covering clauses", timed_result_face_clauses.time, timed_result_face_clauses.bytes))

    # --- Step 4c: Flag SAT formulation over triples ---
    if config.flag_SAT
        log_verbose("Step 4c: Generating flag SAT clauses (optional, controlled by flag_SAT)")
        if !config.flag_triangulation  
            error("flag_SAT set to true but flag_triangulation set to false")
        end
        timed_result_flag_clauses = @timed compute_flag_clauses(S_indices, intersection_matrix) #via Theorem 3.1 of https://arxiv.org/abs/2411.12945
        flag_clauses = timed_result_flag_clauses.value
        append!(cnf, flag_clauses)
        push!(step_stats, StepStat("Generate flag clauses", timed_result_face_clauses.time, timed_result_face_clauses.bytes))
    end

    # --- Step 5: Solving ---
    log_verbose("Step 5: Handing SAT problem to solver...")
    log_verbose("      Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")

    # Determine solver
    active_solver = config.solver
    
    log_verbose("      Using solver: $active_solver")

    timed_solve_result = @timed begin
        if config.incremental_solving
            # setup_run guarantees solver == "cadical" here and has disabled
            # parallel solving (the two modes are mutually exclusive).
            log_verbose("      Incremental solving enabled with $(nthreads()) threads.")
            solve_cadical_incremental(cnf, P, S_indices, dim, config, show_running_updates, log_verbose)
        elseif config.parallel_split_solving
            log_verbose("      Parallel solving enabled with $(nthreads()) threads.")
            log_verbose("      Solver is $(active_solver)")
            solve_parallel(cnf, P, S_indices, internal_faces_set, config, show_running_updates)
        elseif active_solver == "picosat"
            solve_picosat(cnf, P, S_indices, config, show_running_updates, Atomic{Bool}(false))
        else
            solve_cadical_standard(cnf, P, S_indices, config, show_running_updates, Atomic{Bool}(false))
        end
    end

    if show_running_updates
        ghost_print("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices... solving...")
    end

    solution_simplices, 
    first_solution_simplices, 
    number_of_triangulations_found, 
    number_of_regular_triangulations_found,
    number_of_flag_triangulations_found,
    number_of_quadratic_triangulations_found = timed_solve_result.value

    push!(step_stats, StepStat("Solve SAT problem", timed_solve_result.time, timed_solve_result.bytes))
    log_verbose("-> SAT solver finished. Step 5 complete.")

    log_verbose("\n$(number_of_triangulations_found) valid triangulation(s) found.")
    if config.regular
        log_verbose("\n$(number_of_regular_triangulations_found) valid regular triangulation(s) found.")
        if config.flag_triangulation
            log_verbose("\n$(number_of_quadratic_triangulations_found) valid quadratic triangulation(s) found.")
        end
    elseif config.flag_triangulation
        log_verbose("\n$(number_of_flag_triangulations_found) valid flag triangulation(s) found.")
    end

    if !isempty(first_solution_simplices)
        log_verbose("\nDisplaying first solution:")
        for s in first_solution_simplices
            log_verbose(s, is_display=true)
        end
        # Validate that #simplices is right
        log_verbose("\nValidating first solution volume-wise:")
        poly = polyhedron(vrep(initial_vertices), CDDLib.Library(:exact))
        if factorial(dim)*volume(poly) != size(first_solution_simplices, 1)
            println(volume(poly))
            println(size(first_solution_simplices,1))
            println()
            error("Found triangulation has wrong number of simplices... please contact us...")
        end
        #TODO: garbage collection of the polyhedron here?
    end

    if isempty(first_solution_simplices)
        log_verbose("\nNo triangulation found for this polytope.")
    end

    # --- Step 6: Plotting ---
    # Invoked by setting 
    #       plot = true
    if config.plot
        log_verbose("\nStep 6: Plotting result..")

        if isempty(first_solution_simplices)
            @error("Cannot plot, no triangulation found")
        else
             plot(initial_vertices, dim, first_solution_simplices)
        end
        log_verbose("-> Plotting complete. Step 6 complete.")
    end

    # --- Summary ---
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
    # NOTE: the per-polytope one-line summary string ("minimal log") used to be
    # built here and stored in every TriangulationResult. It was never printed
    # anywhere, but retaining one string per polytope adds up over runs with
    # millions of polytopes, so it was removed from the result type.

    return TriangulationResult(solution_simplices, number_of_triangulations_found, number_of_regular_triangulations_found, number_of_flag_triangulations_found, number_of_quadratic_triangulations_found, total_time, step_stats)
end

function print_initial_summary(config::Config, n_polytopes::Int, stream::IO)
    println(stream, "Run started at:                      $(Dates.format(now(), "HH:MM:SS"))")
    println(stream, "Number of threads:                   $(nthreads())")
    println(stream, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
    println(stream, "Solver:                              $(config.solver)")
    println(stream, "Parallel Solving:                    $(config.parallel_split_solving)")
    println(stream, "Incremental Solving:                 $(config.incremental_solving)")
    println(stream, "Checking for full dimensionality:    $(config.check_full_dimensionality)")
    println(stream, "Intersection backend selected:       $(config.intersection_backend)")
#    println(stream, "Validation enabled:                  $(config.validate)")
    println(stream, "Number of polytopes found:           $(n_polytopes)")
    println(stream, "Restricting to unimodular simplices: $(config.unimodular)")
    println(stream, "Looking for regular triangulations:  $(config.regular)")
    println(stream, "Looking for flag triangulations:     $(config.flag_triangulation)")
    println(stream, "Using Normaliz:                      $(config.use_normaliz)")
    println(stream, "")
end

function run_processing(polytopes::Vector{Matrix{Int}}, config::Config, log_stream)
    number_of_polytopes = length(polytopes)
    components_str = lowercase(replace(config.terminal_output, " " => ""))

    show_initial = occursin("initial", components_str) || occursin("all", components_str)
    show_running = occursin("running", components_str) || occursin("all", components_str)
    show_table   = occursin("table", components_str)   || occursin("all", components_str)
    show_final   = occursin("final", components_str)   || occursin("all", components_str)

    if show_initial && isnothing(log_stream)
        term_summary_buf = IOBuffer()
        print_initial_summary(config, number_of_polytopes, term_summary_buf)
        print(stdout, String(take!(term_summary_buf)))
    end

    # Aggregate counters across all polytopes
    t_start_run = time()
    triangulatable = 0
    regularly_triangulatable = 0
    flag_triangulatable = 0
    quadratic_triangulatable = 0

    total_number_of_triangulations_found = 0
    total_number_of_regular_triangulations_found = 0
    total_number_of_flag_triangulations_found = 0
    total_number_of_quadratic_triangulations_found = 0

    global_step_stats = Dict{String, StatAggregator}()
    step_order = String[]
    all_results = Vector{TriangulationResult}()
    # With return_triangulations="none", per-polytope results are not retained:
    # over runs with millions of polytopes, one TriangulationResult per polytope
    # (log string, step stats and possibly a stored triangulation) adds up to
    # gigabytes. Aggregate counters and the log file are unaffected.
    keep_individual_results = config.return_triangulations != "none"

    for (i, P) in enumerate(polytopes)
        r = process_polytope(P, i, number_of_polytopes, config, show_running, log_stream)
        if keep_individual_results
            push!(all_results, r)
        end

        # Native objects (CDD-exact polyhedra and H-representations, the
        # regularity-LP polyhedra, Normaliz cones, PicoSAT instances) are
        # freed by finalizers, and their memory is invisible to Julia's GC
        # heuristics. Crucially, these objects live long enough to be
        # promoted to the old generation, where *incremental* collections
        # (GC.gc(false)) never finalize them -- only full collections do.
        # A full collection on the small live set is cheap (tens of ms)
        # relative to the work done per interval.
        if i % 250 == 0
            GC.gc()
            # Opt-in memory telemetry (set UNITRISAT_MEM_LOG=1): compare the
            # slope of gc_live (Julia-side retention) against maxrss (native
            # + fragmentation) to attribute any remaining growth.
            if haskey(ENV, "UNITRISAT_MEM_LOG") && !isnothing(log_stream)
                println(log_stream, "[memlog] polytope $i: gc_live=$(round(Base.gc_live_bytes()/1024^2; digits=1)) MiB, maxrss=$(round(Sys.maxrss()/1024^2; digits=1)) MiB")
                flush(log_stream)
            end
        end

        if !isnothing(log_stream); flush(log_stream); end

        # Aggregate logical classification properties
        if r.number_of_triangulations_found > 0;         triangulatable += 1; end
        if r.number_of_regular_triangulations_found > 0; regularly_triangulatable += 1; end
        if r.number_of_flag_triangulations_found > 0;    flag_triangulatable += 1; end
        if r.number_of_quadratic_triangulations_found > 0; quadratic_triangulatable += 1; end

        # Aggregate quantitative total tallies
        total_number_of_triangulations_found           += r.number_of_triangulations_found
        total_number_of_regular_triangulations_found   += r.number_of_regular_triangulations_found
        total_number_of_flag_triangulations_found      += r.number_of_flag_triangulations_found
        total_number_of_quadratic_triangulations_found += r.number_of_quadratic_triangulations_found

        # Update profiling analytics maps
        for stat in r.step_stats
            agg = get!(global_step_stats, stat.name) do
                push!(step_order, stat.name)
                StatAggregator()
            end
            agg.total_time  += stat.duration_s
            agg.max_time     = max(agg.max_time, stat.duration_s)
            agg.total_alloc += stat.alloc_bytes
            agg.max_alloc    = max(agg.max_alloc, stat.alloc_bytes)
            agg.count       += 1
        end

        if show_running
            elapsed_time = time() - t_start_run
            eta_str = format_duration((elapsed_time / i) * (number_of_polytopes - i))
            
            # Dynamic suffix showing flag/regular stats depending on configuration settings
            sub_type_str = config.regular ? "\nRegularly Triangulatable:    \u001b[32m$(regularly_triangulatable)\u001b[0m" : ""
            if config.flag_triangulation
                sub_type_str *= "\nFlag Triangulatable:         \u001b[32m$(flag_triangulatable)\u001b[0m"
                if config.regular
                    sub_type_str *= "\nQuadratic Triangulatable:    \u001b[32m$(quadratic_triangulatable)\u001b[0m"
                end
            end

            ghost_print("""\n
            Elapsed Time:                 $(format_duration(elapsed_time))
            Estimated Time Left:          $eta_str
            Triangulatable:               \u001b[32m$triangulatable\u001b[0m$sub_type_str
            Non-Triangulatable:           \u001b[31m$(i - triangulatable)\u001b[0m
            """)
        end
    end

    total_time_run = time() - t_start_run

    # Generate analytical tables and final diagnostic summaries
    stats_table_str = ""
    if show_table && !isempty(global_step_stats)
        buf = IOBuffer()
        print_stats_table!(buf, global_step_stats, step_order)
        stats_table_str = String(take!(buf))
    end

    if show_final
        avg_solutions_str = ""
        if config.find_all
            # Calculate the target numerical output matching requested settings
            num_sol = config.regular ? 
                     (config.flag_triangulation ? total_number_of_quadratic_triangulations_found : total_number_of_regular_triangulations_found) : 
                     (config.flag_triangulation ? total_number_of_flag_triangulations_found : total_number_of_triangulations_found)
            avg_solutions_str = @sprintf("Average Target Solutions/Poly: %.2f\n", num_sol / number_of_polytopes)
        end
        
        reg_str  = config.regular ? "\nRegularly Triangulatable:       \u001b[32m$regularly_triangulatable\u001b[0m" : ""
        flag_str = config.flag_triangulation ? "\nFlag Triangulatable:            \u001b[32m$flag_triangulatable\u001b[0m" : ""
        quad_str = (config.regular && config.flag_triangulation) ? "\nQuadratic Triangulatable:       \u001b[32m$quadratic_triangulatable\u001b[0m" : ""

        summary_core_str = """
        Run finished: $(Dates.format(now(), "HH:MM:SS"))                                      
                                                                                               
        ----------------------------------------                                               
        Run Summary                                                                           
        ----------------------------------------                                               
        Total Polytopes Processed:     $number_of_polytopes$reg_str$flag_str$quad_str
        Triangulatable:                \u001b[32m$triangulatable\u001b[0m
        Non-Triangulatable:            \u001b[31m$(number_of_polytopes - triangulatable)\u001b[0m
        $(avg_solutions_str)Total Run Time:                $(format_duration(total_time_run))
        ----------------------------------------
        \n"""
        
        print(summary_core_str)
        if show_table; print(stats_table_str); end

        if !isnothing(log_stream)
            print(log_stream, strip_ansi(summary_core_str))
            if show_table; print(log_stream, strip_ansi(stats_table_str)); end
            flush(log_stream)
        end
    end

    return RunResult(
        all_results, 
        triangulatable, 
        regularly_triangulatable, 
        flag_triangulatable, 
        quadratic_triangulatable,
        total_number_of_triangulations_found, 
        total_number_of_regular_triangulations_found,
        total_number_of_flag_triangulations_found,
        total_number_of_quadratic_triangulations_found,
        total_time_run
    )
end

# Private helper to decouple large tabular printing tasks from core processing control-flow
function print_stats_table!(buf::IOBuffer, global_step_stats::Dict{String, StatAggregator}, step_order::Vector{String})
    println(buf, "\n", @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s | %-12s",
                    "Step Name", "Total Time", "Avg Time", "Max Time", "Avg Memory", "Max Memory"))
    println(buf, "-"^108)
    for name in step_order
        stat = global_step_stats[name]
        println(buf, @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s | %-12s",
                    name, format_duration(stat.total_time), @sprintf("%.3f s", stat.total_time / stat.count),
                    @sprintf("%.3f s", stat.max_time), format_bytes(stat.total_alloc / stat.count), format_bytes(stat.max_alloc)))
    end
end

function setup_run( polytopes::Vector{Matrix{Int}},
                    intersection_backend::String="cpu",
                    unimodular::Bool=true, regular::Bool=false, flag_triangulation::Bool=false, flag_SAT=false,
                    find_all::Bool=false, log_file::String="",
                    terminal_output::String="final",
                    validate::Bool=false,
                    plot::Bool=false,
                    use_normaliz::Bool=false,
                    return_triangulations::String="first",
                    solver::String="picosat",
                    incremental_solving::Bool=false,
                    circuit_intersection_clauses::Bool=false,
                    check_full_dimensionality::Bool=false,
                    parallel_split_solving::Bool=true)

    if intersection_backend == "gpu"
        # Scan the whole run; warn only if some polytope can actually overflow.
        # Vertex coordinates bound the lattice points, so no enumeration needed.
        Precision.check_gpu_precision(polytopes)
    end

    if isempty(polytopes)
        @warn("No polytopes provided to setup_run.")
        return RunResult(Vector{TriangulationResult}(), 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    end

    cadical_available = Sys.islinux() || Sys.isapple()

    if solver in ["cadical"] && !cadical_available
        @warn("CaDiCaL is only available on Linux and Mac atm. Falling back to PicoSat")
        solver="picosat"
    end

    if incremental_solving
        if solver != "cadical"
            if cadical_available
                @warn("Incremental solving requires CaDiCaL; switching solver from \"$solver\" to \"cadical\".")
                solver = "cadical"
            else
                @warn("Incremental solving requires CaDiCaL, which is unavailable on this platform. Disabling incremental solving.")
                incremental_solving = false
            end
        end
        if incremental_solving && parallel_split_solving
            @info("Incremental solving and parallel solving are mutually exclusive; parallel solving is disabled for this run.")
            parallel_split_solving = false
        end
    end

    #= We got rid of d4 because we don't need it
    if solver == "d4" && !Sys.islinux()
        @warn("d4 is only available on Linux. Falling back to PicoSat")
        solver="picosat"
    end

    if solver in ["picosat", "d4"] && incremental_solving
        @warn("Incremental solving is only supported by CaDiCaL, not PicoSat or d4. We keep incremental solving true. The solver will be passed the simplified formula, but no further clauses will be added.")
    end

    if solver == "d4" && !find_all
        @warn("The d4 solver only supports finding all triangulations, but the find_all flag is not set. Falling back to PicoSat.")
        solver = "picosat"
    end
    
    if solver == "picosat" && incremental_solving
    	@warn("Incremental solving is only supported by CaDiCaL, not PicoSat or d4. We keep incremental solving true. The solver will be passed the simplified formula, but no further clauses will be added.")
	end
    =#
    
    if !(return_triangulations in ("first", "all", "none"))
        @warn("Unknown return_triangulations=\"$return_triangulations\"; expected \"first\", \"all\" or \"none\". Falling back to \"first\".")
        return_triangulations = "first"
    end

    if use_normaliz && !Normaliz_available[]
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

    config = Config(terminal_output, unimodular, intersection_backend, regular, flag_triangulation, flag_SAT, find_all, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses, check_full_dimensionality, parallel_split_solving)
    log_stream = nothing
    
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
    
        return run_processing(polytopes, config, log_stream)
    finally
        if !isnothing(log_stream)
            close(log_stream)
        end
    end
end

# Various triangulate entry points. We allow differing inputs for user convenience.
# via the extension /ext/UniTriSatOscarExt.jl this should work for Oscar polyhedra as well.

function triangulate(   vmatrix::Matrix{Int};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        flag_triangulation::Bool=false,
                        flag_SAT::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="final",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        circuit_intersection_clauses::Bool=false,
                        check_full_dimensionality::Bool=false,
                        parallel_split_solving::Bool=true)

     return setup_run([vmatrix], intersection_backend, unimodular, regular, flag_triangulation, flag_SAT, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses,  check_full_dimensionality, parallel_split_solving)
end

function triangulate(   vmatrices::Vector{Matrix{Int}};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        flag_triangulation::Bool=false,
                        flag_SAT::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="final",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        circuit_intersection_clauses::Bool=false,
                        check_full_dimensionality::Bool=false,
                        parallel_split_solving::Bool=true)

    return setup_run(vmatrices, intersection_backend, unimodular, regular, flag_triangulation, flag_SAT, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses, check_full_dimensionality, parallel_split_solving)
end

function triangulate(   polytope::Polyhedron;
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        flag_triangulation::Bool=false,
                        flag_SAT::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="final",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        circuit_intersection_clauses::Bool=false,
                        check_full_dimensionality::Bool=false,
                        parallel_split_solving::Bool=true)

    vmatrix = _convert_polyhedron_to_vmatrix(polytope)
    if isempty(vmatrix)
        @error("Could not process a single polytope")
        return nothing
    end
    return setup_run([vmatrix], intersection_backend, unimodular, regular, flag_triangulation, flag_SAT, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses,  check_full_dimensionality, parallel_split_solving)
end

function triangulate(   polytopes::Vector{Polyhedron};
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        flag_triangulation::Bool=false,
                        flag_SAT::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="final",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        circuit_intersection_clauses::Bool=false,
                        check_full_dimensionality::Bool=false,
                        parallel_split_solving::Bool=true)

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
    return setup_run(vmatrices, intersection_backend, unimodular, regular, flag_triangulation, flag_SAT, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses,  check_full_dimensionality, parallel_split_solving)
end

function triangulate(   path_to_polytopes::String;
                        intersection_backend::String="cpu",
                        unimodular::Bool=true,
                        regular::Bool=false,
                        flag_triangulation::Bool=false,
                        flag_SAT::Bool=false,
                        find_all::Bool=false,
                        log_file::String="",
                        terminal_output::String="final",
                        validate::Bool=false,
                        plot::Bool=false,
                        use_normaliz::Bool=false,
                        return_triangulations::String="first",
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        circuit_intersection_clauses::Bool=false,
                        check_full_dimensionality::Bool=false,
                        parallel_split_solving::Bool=true)

    local polytopes
    full_path = abspath(path_to_polytopes)

    try
        # Try to read the polytopes using the now-accessible path
        polytopes = read_polytopes_from_file(full_path) 
        
        if isempty(polytopes)
            @error("Error: No polytopes loaded from '$full_path'.")
            return Vector{Vector{Matrix{Int}}}[]
        end
        
    catch e
        @error("Error loading polytopes from '$full_path': '$e'") 
        return Vector{Vector{Matrix{Int}}}[]
    end
    return setup_run(polytopes, intersection_backend, unimodular, regular, flag_triangulation, flag_SAT, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, circuit_intersection_clauses,  check_full_dimensionality, parallel_split_solving)
end

end

module UniTriSat

export triangulate

using Combinatorics
using LinearAlgebra
using Polyhedra
using Dates
using Printf
using Base.Threads
using TOML
using Random
using CDDLib
using StaticArrays
using AbstractAlgebra

include("structs.jl")
using .Structs

# include the rest of the modules

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
    #
    # Checks if our polytope is full-dimensional. 
    # When it's not, we compute its Hermite Normal Form and find a lattice-equivalent polytope 
    # in a lower ambient dimension.

    # If the flag is active and there's a dimension mismatch
    v = vrep(initial_vertices)
    poly = polyhedron(v, CDDLib.Library(:exact))
    # Computing a lattice-preserving projection
    if config.check_full_dimensionality && Polyhedra.dim(poly) < dim 
        #display(initial_vertices)
        log_verbose("Finding an appropriate projection to remove excess ambient dimensions")
        # We use Hermite Normal Form to compute a lattice-equivalent polytope in a lower-dimensional space
        initial_vertices = full_dimensional_lattice_projection(initial_vertices)
        dim = size(initial_vertices,2)

        v = vrep(initial_vertices)
        poly = polyhedron(v, CDDLib.Library(:exact))
        dim_poly = Polyhedra.dim(poly)
        # TODO: why is this here? Is it excess? 
        if false && dim_poly < dim 
            log_verbose("Polytope is not full-dimensional. Original dimension: $dim, polytope dimension: $dim_poly")
            display(initial_vertices)
            log_verbose("Finding an appropriate projection to remove excess ambient dimensions")
            # We use Hermite Normal Form to compute a lattice-equivalent polytope in a lower-dimensional space
        
            # Shift to the origin
            v0 = initial_vertices[:, 1]
            shifted_vertices = initial_vertices .- v0
        
            # Compute HNF of the edge vectors
            # We transpose because HNF usually works on rows. Later we transpose it back again

            M = matrix(ZZ, transpose(shifted_vertices))

            H, U = hnf_with_transform(M) 
            display(H)
            # Extract non-zero columns
            nz_rows = [row_index for row_index in 1:nrows(H) if !is_zero(H[row_index, :])]
            
            println(nz_rows)    
            # The new vertices are the rows of the submatrix
            projected_coords = [Int64(H[row_index, j]) for row_index in nz_rows, j in 1:size(H, 2)]
        
            # Massaging the output
            # Give new initial vertices and new dimension
            initial_vertices = copy(transpose(projected_coords)) # Have to copy since Julia does weird things when transposing?
            dim = size(initial_vertices,2)

            println(size(initial_vertices, 1))

            v = vrep(initial_vertices)
            poly = polyhedron(v, CDDLib.Library(:exact))

            if Polyhedra.dim(poly) != size(initial_vertices,2)
                error("Projection failed to produce a full-dimensional polytope. This should never happen... if you see this error please open an issue on GitHub or write us some other way...")
            end
            log_verbose("Projected vertices")
            log_verbose(initial_vertices, is_display=true)
        else 
            log_verbose("Polytope is full-dimensional. Continuing with original vertices.")
        end
    end
    

    # --- Step 1: Lattice Points ---
    log_verbose("Step 1: Computing all lattice points...")
    timed_result_lp = @timed compute_lattice_points(initial_vertices, config)
    P = timed_result_lp.value
    push!(step_stats, StepStat("Compute all lattice points", timed_result_lp.time, timed_result_lp.bytes))

    num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    ghost_print("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points...")

    # --- Step 2: Simplices ---
    simplex_search_type = config.unimodular ? "unimodular" : "non-degenerate"
    log_verbose("Step 2: Computing $simplex_search_type $(dim)-simplices...")

    timed_result_simplices = @timed compute_simplices(P, config)
    S_indices = timed_result_simplices.value
    push!(step_stats, StepStat("Compute $simplex_search_type simplices", timed_result_simplices.time, timed_result_simplices.bytes))

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))
    log_verbose("-> Found $num_simplices simplices. Step 2 complete.\n")
    ghost_print("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices...")

    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> No simplices found", run_idx, total_in_run, num_lattice_points, num_simplices)
        return TriangulationResult([], 0, 0, minimal_log, time()-t_start_total, step_stats)
    end

    # --- Step 3: Internal Faces ---
    log_verbose("Step 3: Computing internal faces...")
    timed_result_faces = @timed compute_internal_faces(P, dim)
    internal_faces_set = timed_result_faces.value
    push!(step_stats, StepStat("Compute internal faces", timed_result_faces.time, timed_result_faces.bytes))
    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")

    # --- Step 4: Intersections ---
    if config.incremental_solving
        log_verbose("Step 4a: Computing intersection clauses...")
        timed_result_intersections = @timed compute_intersections_incremental(P, S_indices, internal_faces_set, dim, num_lattice_points)
        intersection_clauses = timed_result_intersections.value
        push!(step_stats, StepStat("Compute intersection clauses", timed_result_intersections.time, timed_result_intersections.bytes))
        append!(cnf, intersection_clauses)
        log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")
    else
        # Non-incremental: Use provided backend logic to find all intersections
        log_verbose("Step 4a: Computing intersecting pairs via hyperplanes...")
        timed_result_intersections = @timed compute_intersections_standard(P, S_indices, dim, config, log_verbose)
        intersection_clauses = timed_result_intersections.value
        push!(step_stats, StepStat("Compute intersecting pairs", timed_result_intersections.time, timed_result_intersections.bytes))
        append!(cnf, intersection_clauses)
        log_verbose("-> Generated $(length(intersection_clauses)) intersection clauses. Step 4 complete.\n")
    end

    # --- Step 4d: Face-Covering Clauses ---
    log_verbose("Step 4b: Generating face-covering clauses...")
    timed_result_face_clauses = @timed compute_face_clauses(S_indices, internal_faces_set, dim)
    face_clauses = timed_result_face_clauses.value
    append!(cnf, face_clauses)
    push!(step_stats, StepStat("Generate face-covering clauses", timed_result_face_clauses.time, timed_result_face_clauses.bytes))

    # --- Step 5: Solving ---
    log_verbose("Step 5: Handing SAT problem to solver...")
    log_verbose("      Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")

    # Determine solver
    active_solver = config.solver
    
    log_verbose("      Using solver: $active_solver")

    timed_solve_result = @timed begin
        if active_solver in ["picosat"]
             solve_picosat(cnf, P, S_indices, config, show_running_updates)
        else
            if config.incremental_solving
                 solve_cadical_incremental(cnf, P, S_indices, dim, config, show_running_updates, log_verbose)
            else
                if config.find_all && config.enable_parallel
                    log_verbose("      Using parallelized Cadical solver...")
                    solve_cadical_parallel(cnf, P, S_indices, config, show_running_updates)
                else
                    solve_cadical_standard(cnf, P, S_indices, config, show_running_updates)
                end
            end
        end
    end
    
    solution_simplices, 
    first_solution_simplices, 
    first_regular_solution_simplices, 
    number_of_triangulations_found, 
    number_of_regular_triangulations_found = timed_solve_result.value

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

    # --- Step 6: Plotting ---
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

function print_initial_summary(config::Config, n_polytopes::Int, stream::IO)
    println(stream, "Run started at:                      $(Dates.format(now(), "HH:MM:SS"))")
    println(stream, "Number of threads:                   $(nthreads())")
    println(stream, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
    println(stream, "Solver:                              $(config.solver)")
    println(stream, "Parallel Solving:                    $(config.enable_parallel)")
    println(stream, "Incremental Solving:                 $(config.incremental_solving)") # TODO: this should reflect if find_all is true or not
    println(stream, "Checking for full dimensionality:    $(config.check_full_dimensionality)")
    println(stream, "Intersection backend selected:       $(config.intersection_backend)")
    println(stream, "Validation enabled:                  $(config.validate)")
    println(stream, "Number of polytopes found:           $(n_polytopes)")
    println(stream, "Restricting to unimodular simplices: $(config.unimodular)")
    println(stream, "Looking for regular triangulations:  $(config.regular)")
    println(stream, "Using Normaliz:                      $(config.use_normaliz)")
    println(stream, "")
end

function run_processing(polytopes::Vector{Matrix{Int}}, config::Config, log_stream)
    number_of_polytopes = length(polytopes)
    components_str = lowercase(replace(config.terminal_output, " " => ""))

    show_initial = occursin("initial", components_str) || occursin("all", components_str)
    show_running = occursin("running", components_str) || occursin("all", components_str)
    show_table = occursin("table", components_str) || occursin("all", components_str)
    show_final = occursin("final", components_str) || occursin("all", components_str)

    if show_initial && isnothing(log_stream)
        term_summary_buf = IOBuffer()
        print_initial_summary(config, number_of_polytopes, term_summary_buf)
        print(stdout, String(take!(term_summary_buf)))
    end

    # the stats to aggregate over all polytopes
    # i.e. triangulatable will be increased by 1 for each triangulatable polytope found
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

        # find results and assign to local vars
        r = process_polytope(P, i, length(polytopes), config, show_running, log_stream)
        number_of_triangulations_found = r.number_of_triangulations_found
        number_of_regular_triangulations_found = r.number_of_regular_triangulations_found
        minimal_log = r.minimal_log
        step_stats = r.step_stats

        push!(all_results, r)

        if !isnothing(log_stream)
             flush(log_stream)
        end

        # adjust counts accordingly
        if number_of_triangulations_found > 0
            triangulatable += 1
        end
        if number_of_regular_triangulations_found > 0
            regularly_triangulatable += 1
        end
        total_number_of_triangulations_found += number_of_triangulations_found
        total_number_of_regular_triangulations_found += number_of_regular_triangulations_found

        # update global stats with the returned step_stats results
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

        # print the live summary
        if show_running
            elapsed_time = time() - t_start_run
            avg_time = elapsed_time / i
            remaining = number_of_polytopes - i
            eta_seconds = avg_time * remaining
            eta_str = format_duration(eta_seconds)

            reg_str = config.regular ? "\nRegularly Triangulatable:     \u001b[32m$(regularly_triangulatable)\u001b[0m" : ""
            live_summary_str = """\n
            Elapsed Time:                 $(format_duration(elapsed_time))
            Estimated Time Left:          $eta_str
            Triangulatable:               \u001b[32m$triangulatable\u001b[0m$reg_str
            Non-Triangulatable:           \u001b[31m$(i - triangulatable)\u001b[0m
            """

            ghost_print(live_summary_str)
        end
        step_stats = nothing
    end

    total_time_run = time() - t_start_run

    avg_solutions_str = ""
    if config.find_all
        num_sol = config.regular ? total_number_of_regular_triangulations_found : total_number_of_triangulations_found
        avg_solutions_str = @sprintf("Average Solutions/Polytope:    %.2f\n", num_sol / number_of_polytopes)
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
    println()

    if show_final
        avg_solutions_str = ""
        if config.find_all
            num_sol = config.regular ? total_number_of_regular_triangulations_found : total_number_of_triangulations_found
            avg_solutions_str = @sprintf("Average Solutions/Polytope:    %.2f\n", num_sol / number_of_polytopes)
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
        $(avg_solutions_str)Total Run Time:                $(format_duration(total_time_run))
        ----------------------------------------


        """
        print(summary_core_str)
    end
    if show_table
        print(stats_table_str)
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
                    solver::String="picosat",
                    incremental_solving::Bool=false,
                    check_full_dimensionality::Bool=false,
                    enable_parallel::Bool=true)

    # Centralized validation logic
    if intersection_backend == "gpu"
        @warn("You have selected the gpu backend. Please note that this backend is subject to overflow errors even for reasonably sized polytopes. Please validate any triangulation found for intersecting simplices and do not trust negative results.")
    end

    if isempty(polytopes)
        @warn("No polytopes provided to setup_run.")
        return RunResult(Vector{TriangulationResult}(), 0, 0, 0, 0, 0.0)
    end

    if solver in ["cadical"] && !Sys.islinux()
        @warn("CaDiCaL is only available on Linux atm. Falling back to PicoSat")
        solver="picosat"
    end

    if solver in ["cadical"] && incremental_solving
        @warn("Incremental solving is only supported by CaDiCaL, not PicoSat")
        incremental_solving = false
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

    config = Config(terminal_output, unimodular, intersection_backend, regular, find_all, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
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
# TODO: allow OSCAR polytopes also

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
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        check_full_dimensionality::Bool=false,
                        enable_parallel::Bool=true)

     return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
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
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        check_full_dimensionality::Bool=false,
                        enable_parallel::Bool=true)

    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
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
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        check_full_dimensionality::Bool=false,
                        enable_parallel::Bool=true)

    vmatrix = _convert_polyhedron_to_vmatrix(polytope)
    if isempty(vmatrix)
        @error("Could not process a single polytope")
        return nothing
    end
    return setup_run([vmatrix], intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
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
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        check_full_dimensionality::Bool=false,
                        enable_parallel::Bool=true)

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
    return setup_run(vmatrices, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
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
                        solver::String="picosat",
                        incremental_solving::Bool=false,
                        check_full_dimensionality::Bool=false,
                        enable_parallel::Bool=true)

    local polytopes
    try
        polytopes = read_polytopes_from_file(path_to_polytopes)
        if isempty(polytopes); @error("Error: No polytopes loaded from '$path_to_polytopes'."); return Vector{Vector{Matrix{Int}}}[]; end
        catch e
        @error("Error loading polytopes from '$path_to_polytopes': '$e'")
        return Vector{Vector{Matrix{Int}}}[]
    end
    return setup_run(polytopes, intersection_backend, unimodular, regular, find_all, log_file, terminal_output, validate, plot, use_normaliz, return_triangulations, solver, incremental_solving, check_full_dimensionality, enable_parallel)
end

end

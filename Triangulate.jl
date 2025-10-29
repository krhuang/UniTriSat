module Triangulate

export triangulate

using Oscar: convex_hull, lattice_points
using Combinatorics
using LinearAlgebra
using Polyhedra # Importiert für die Konvertierung
using PicoSAT
using Dates
using Printf
using Base.Threads
using TOML
using Random
# using Normaliz 
# Could use Oscar or Normaliz as the backend to find lattice points?
# plotting functions
include("plotting_utils.jl") # TODO: do this better; as a module rather than as a script
# --- Conditional Package Inclusion ---
const CUDA_PACKAGES_LOADED = Ref(false)
try
    using CUDA, StaticArrays, CUDA.Adapt
    CUDA_PACKAGES_LOADED[] = true
catch
end

# ... (Rest der include-Anweisungen für GPU-Backends bleibt unverändert) ...
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_3d.jl")
    include("Intersection_backends/gpu_intersection_3d.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_4d.jl")
    include("Intersection_backends/gpu_intersection_4d.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_5d.jl")
    include("Intersection_backends/gpu_intersection_5d.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_6d.jl")
    include("Intersection_backends/gpu_intersection_6d.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_3d_floats.jl")
    include("Intersection_backends/gpu_intersection_3d_floats.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_4d_floats.jl")
    include("Intersection_backends/gpu_intersection_4d_floats.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_5d_floats.jl")
    include("Intersection_backends/gpu_intersection_5d_floats.jl")
end
if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_6d_floats.jl")
    include("Intersection_backends/gpu_intersection_6d_floats.jl")
end

include("Intersection_backends/cpu_intersection.jl")

# --- Angepasste Strukturen ---

"""
Speichert die gemessene Performance für einen einzelnen Verarbeitungsschritt.
"""
struct StepStats
    name::String
    duration_s::Float64
    alloc_bytes::Int64
end

"""
Speichert das Gesamtergebnis der Verarbeitung eines einzelnen Polytops.
"""
struct ProcessResult
    id::Int
    num_solutions_found::Int
    total_time::Float64
    num_lattice_points::Int
    num_simplices_considered::Int
    verbose_log::String
    minimal_log::String
    first_solution_simplices::Vector{Matrix{Int}}
    step_stats::Vector{StepStats}
end

# --- Hilfsfunktionen ---

function format_duration(total_seconds::Float64)
    total_seconds_int = floor(Int, total_seconds)
    h = total_seconds_int ÷ 3600
    rem_seconds = total_seconds_int % 3600
    m = rem_seconds ÷ 60
    s = rem_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function format_bytes(b::Real)
    if b > 1024^3
        return @sprintf("%.2f GiB", b / 1024^3)
    elseif b > 1024^2
        return @sprintf("%.2f MiB", b / 1024^2)
    elseif b > 1024
        return @sprintf("%.2f KiB", b / 1024)
    else
        return @sprintf("%d B", b)
    end
end

function read_polytopes_from_file(filepath::String)
    polytopes = Vector{Matrix{Int}}()
    current_vertices = Vector{Vector{Int}}()
    function process_buffered_vertices()
        if !isempty(current_vertices)
            push!(polytopes, vcat(current_vertices'...))
            empty!(current_vertices)
        end
    end
    for line in eachline(filepath)
        line = strip(line)
        if isempty(line) || startswith(line, "#"); process_buffered_vertices(); continue; end
        vertex_pattern = r"\[([^\[\]]+)\]"
        if startswith(line, "[[")
            process_buffered_vertices()
            vertices_new_format = Vector{Vector{Int}}()
            for m in eachmatch(vertex_pattern, line)
                push!(vertices_new_format, parse.(Int, split(m.captures[1], ",")))
            end
            if !isempty(vertices_new_format); push!(polytopes, vcat(vertices_new_format'...)); end
        else
            try; push!(current_vertices, parse.(Int, split(line))); catch e; @warn "Skipping malformed line: $line. Error: $e"; end
        end
    end
    process_buffered_vertices()
    return polytopes
end

"""
NEU: Konvertiert ein Polyhedra.jl Polyhedron-Objekt in eine Matrix{Int} 
von Scheitelpunkten (V-Rep), wie sie vom Rest des Skripts erwartet wird.
Nimmt an, dass die Scheitelpunkte ganzzahlig sind.
"""
function _convert_polyhedron_to_vmatrix(p::Polyhedron)
    # vlist(p) gibt einen Iterator von Vektoren (Punkten) zurück
    # Wir müssen sicherstellen, dass sie als Ints und als Zeilenvektoren in einer Matrix landen
    try
        # vcat(vektor'...): Konvertiert jeden (Spalten-)Vektor zu einem Zeilenvektor und kettet sie vertikal
        return vcat([Int.(v)' for v in vlist(p)]...)
    catch e
        @error("Error converting Polyhedron object to Matrix{Int}: $e")
        return Matrix{Int}(undef, 0, 0) # Leere Matrix zurückgeben
    end
end


# for logs
function update_line(message::String)
    print(stdout, "\r" * message * "\u001b[K");
    flush(stdout)
end


# --- Presolve functions ---

function lattice_points_via_Oscar(vertices::Matrix{Int})
    polytope = convex_hull(vertices)
    LP = lattice_points(polytope) 
    dims = size(LP)
    nrows = dims[1] 
    ncols = size(LP[1])[1]
    julia_matrix_LP = [BigInt(LP[i][j]) for i in 1:nrows, j in 1:ncols]
    return julia_matrix_LP
end

function all_simplices(lattice_points::Matrix{BigInt}; only_unimodular::Bool=false)
    n, d = size(lattice_points)
    simplex_indices = Vector{NTuple{d+1, Int}}()
    if n < d + 1
        return simplex_indices
    end

    for inds in combinations(1:n, d + 1)
        p0 = lattice_points[inds[1], :]
        M = vcat([(lattice_points[inds[i], :] - p0)' for i in 2:(d + 1)]...)
        det_val = det(M)
        if det_val != 0 && (!only_unimodular || abs(det_val) == 1)
            push!(simplex_indices, Tuple(inds))
        end
    end
    return simplex_indices
end

function internal_faces(vertices::Matrix{BigInt}, dim::Int)
    n = size(vertices, 1)
    if n < dim
        return Set{NTuple{dim, Int}}()
    end

    poly = Polyhedra.polyhedron(vrep(vertices))
    hr = hrep(poly)
    planes = collect(halfspaces(hr))
    potential_faces = collect(combinations(1:n, dim))
    next_idx = Threads.Atomic{Int}(1)

    tasks = [
        Threads.@spawn begin
            local_faces = Set{NTuple{dim, Int}}()
            while true
                i = Threads.atomic_add!(next_idx, 1)
                if i > length(potential_faces)
                    break
                end
                face_indices = potential_faces[i]
                face_points = vertices[collect(face_indices), :]
                on_boundary = any(plane -> all(iszero, face_points * plane.a .- plane.β), planes)
                if !on_boundary
                    push!(local_faces, Tuple(sort(collect(face_indices))))
                end
            end
            local_faces
        end
        for _ in 1:nthreads()
    ]
    return union(fetch.(tasks)...)
end

"""
Konfigurationsstruktur für den Triangulierungslauf.
GEÄNDERT: source_file entfernt.
GEÄNDERT: terminal_output ist jetzt ein String.
"""
mutable struct Config
    terminal_output::String # WAR: Bool
    only_unimodular::Bool
    intersection_backend::String
    find_all::Bool
    validate::Bool
    plotter::String
end

# --- Main Processing Functions ---

"""
Verarbeitet ein einzelnes Polytop.
GEÄNDERT: 'dim' wird jetzt hier zuverlässig bestimmt und nicht mehr übergeben.
GEÄNDERT: Nimmt 'show_running_updates::Bool' entgegen, um Terminal-Updates zu steuern.
"""
function process_polytope(initial_vertices::Matrix{Int}, run_idx::Int, total_in_run::Int, config::Config, show_running_updates::Bool)
    # WICHTIG: dim wird hier pro Polytop bestimmt
    dim = size(initial_vertices, 2)
    
    buf = IOBuffer()
    step_stats = Vector{StepStats}()
    t_start_total = time_ns()
    validation_status = :not_run

    function log_verbose(msg...; is_display::Bool=false)
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = if is_display; sprint(show, "text/plain", msg[1]); else; join(msg, " "); end
        full_msg = "[$timestamp] " * s_msg
        println(buf, full_msg)
    end
    
    log_verbose("Processing $(dim)D Polytope #$run_idx") # 'dim' ist jetzt lokal
    log_verbose("Initial vertices provided:")
    log_verbose(initial_vertices, is_display=true)

    log_verbose("Step 1: Finding all lattice points...")
    timed_result_lp = @timed lattice_points_via_Oscar(initial_vertices)
    P = timed_result_lp.value
    push!(step_stats, StepStats("Find all lattice points", timed_result_lp.time, timed_result_lp.bytes))
    
    num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if show_running_updates # GEÄNDERT
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points...")
    end
    
    simplex_search_type = config.only_unimodular ? "unimodular" : "non-degenerate"
    log_verbose("Step 2: Searching for $simplex_search_type $(dim)-simplices...")
    
    timed_result_simplices = @timed all_simplices(P, only_unimodular=config.only_unimodular)
    S_indices = timed_result_simplices.value
    push!(step_stats, StepStats("Find all $simplex_search_type simplices", timed_result_simplices.time, timed_result_simplices.bytes))

    num_simplices_found = length(S_indices)
    log_verbose("-> Found $num_simplices_found simplices. Step 2 complete.\n")
    if show_running_updates # GEÄNDERT
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices_found...")
    end
    
    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        msg = "no $simplex_search_type simplices exist"
        verbose_log = String(take!(buf)) * "\nNo simplices available. Cannot proceed."
        return ProcessResult(run_idx, 0, total_time, num_lattice_points, num_simplices_found, verbose_log, msg, [], step_stats)
    end
    
    log_verbose("Step 3: Computing internal faces...")
    
    # 'dim' wird hier korrekt verwendet
    timed_result_faces = @timed internal_faces(P, dim)
    internal_faces_set = timed_result_faces.value
    push!(step_stats, StepStats("Compute internal faces", timed_result_faces.time, timed_result_faces.bytes))

    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))

    log_verbose("Step 4a: Generating intersection clauses...")
    
    timed_result_intersections = @timed let n_simplices = num_simplices
        intersect_func = nothing
        use_gpu = false

        if config.intersection_backend == "gpu"
            # 'dim' wird hier korrekt verwendet, um das Backend auszuwählen
            if dim == 3 && isdefined(Main, :GPUIntersection3D)
                log_verbose("      Using 3D GPU backend...")
                intersect_func = () -> Main.GPUIntersection3D.get_intersecting_pairs_gpu(P, S_indices)
                use_gpu = true
            elseif dim == 4 && isdefined(Main, :GPUIntersection4D)
                log_verbose("      Using 4D GPU backend...")
                intersect_func = () -> Main.GPUIntersection4D.get_intersecting_pairs_gpu_4d(P, S_indices)
                use_gpu = true
            elseif dim == 5 && isdefined(Main, :GPUIntersection5D)
                log_verbose("      Using 5D GPU backend...")
                intersect_func = () -> Main.GPUIntersection5D.get_intersecting_pairs_gpu_5d(P, S_indices)
                use_gpu = true
            elseif dim == 6 && isdefined(Main, :GPUIntersection6D)
                log_verbose("      Using 6D GPU backend...")
                intersect_func = () -> Main.GPUIntersection6D.get_intersecting_pairs_gpu_6d(P, S_indices)
                use_gpu = true
            end
        end
        if use_gpu && !isnothing(intersect_func)
            intersect_func()
        else
            if !(config.intersection_backend in ["cpu", "gpu", nothing])
                @warn("I do not know config.intersection_backend '$(config.intersection_backend)'. Falling back to CPU.")
                log_verbose("      WARNING: I do not know config.intersection_backend '$(config.intersection_backend)'. Falling back to CPU.")
            end
            if !(dim in [3,4,5,6]) && config.intersection_backend == "gpu"
                @warn("GPU backend for $(dim)D not available. Falling back to CPU.")
                log_verbose("      WARNING: GPU backend for $(dim)D not available. Falling back to CPU.")
            end
            log_verbose("      Using CPU backend.")
            CPUIntersection.get_intersecting_pairs_cpu_generic(P, S_indices)
        end
    end
    
    intersection_clauses = timed_result_intersections.value
    push!(step_stats, StepStats("Generate intersection clauses", timed_result_intersections.time, timed_result_intersections.bytes))
    append!(cnf, intersection_clauses)
    log_verbose("-> Found $(length(intersection_clauses)) intersection clauses. Step 4a complete.\n")

    log_verbose("Step 4b: Generating face-covering clauses...")
    face_dim = dim # 'dim' wird hier korrekt verwendet

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
    push!(step_stats, StepStats("Generate face-covering clauses", timed_result_face_clauses.time, timed_result_face_clauses.bytes))
    log_verbose("-> Found $(length(face_clauses)) face-covering clauses. Step 4b complete.\n")
    
    log_verbose("Step 5: Handing SAT problem to solver...");
    log_verbose("      Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")
    if show_running_updates # GEÄNDERT
        update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices_found solving...")
    end
    
    solutions = []
    num_solutions = 0
    solver_func = PicoSAT

    timed_solve_result = @timed if !config.find_all
        solution = solver_func.solve(cnf)
        if solution isa Vector{Int}
            num_solutions = 1
            push!(solutions, solution)
        end
    else
        for solution in solver_func.itersolve(cnf)
            num_solutions += 1
            push!(solutions, solution)
        end
    end
    
    push!(step_stats, StepStats("Solve SAT problem", timed_solve_result.time, timed_solve_result.bytes))
    log_verbose("-> SAT solver finished. Step 5 complete.")
    
    first_sol_indices = Int[]
    if num_solutions > 0
        first_sol_indices = findall(l -> l > 0, first(solutions))
    end

    if config.validate && num_solutions > 0
        log_verbose("\nStep 5a: Validating solution (not yet implemented)...")
        timed_validation = @timed begin
            validation_status = :passed
            #TODO implement validation
        end
        push!(step_stats, StepStats("Validation", timed_validation.time, timed_validation.bytes))

        if validation_status == :passed
            log_verbose("  VALIDATION SUCCESSFUL: No intersections found among solution simplices.")
        else
            @error("Valitdation failed! Initial vertices where: '$initial_vertices'")
        end
        log_verbose("-> Validation complete. Step 5a complete.")
    end

    total_time = (time_ns() - t_start_total) / 1e9
    log_verbose("\n$(num_solutions) valid triangulation(s) found.")
    
    first_solution_simplices = Vector{Matrix{Int}}()
    if num_solutions > 0 && !isempty(first_sol_indices)
        first_solution_simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in first_sol_indices]
    end

    if !isempty(first_solution_simplices)
        log_verbose("\nDisplaying first valid triangulation:");
        for s in first_solution_simplices
            log_verbose(s, is_display=true)
        end
    end
    
    if !isempty(config.plotter)
        log_verbose("\nStep 6: Plotting results...")
        # 'dim' wird hier korrekt für die Plot-Logik verwendet
        if dim == 3
             temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(config.plotter, first_solution_simplices)); close(temp_io); run(`$(config.plotter) $(temp_path)`); finally rm(temp_path, force=true); end
        elseif dim == 4
            # ... (Plotting-Code für 4D) ...
        elseif dim == 5
            # ... (Plotting-Code für 5D) ...
        elseif dim == 6
            # ... (Plotting-Code für 6D) ...
        end
        
        log_verbose("-> Plotting complete. Step 6 complete.")
    end

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
    if num_solutions > 0
        result_str = @sprintf("\u001b[32mfound %d solution(s)\u001b[0m in %.2f s", num_solutions, total_time)
    else
        result_str = @sprintf("\u001b[31mno solution exists\u001b[0m, searched for %.2f s", total_time)
    end
    minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> %s", run_idx, total_in_run, num_lattice_points, num_simplices_found, result_str)
    
    return ProcessResult(run_idx, num_solutions, total_time, num_lattice_points, num_simplices_found, String(take!(buf)), minimal_log, first_solution_simplices, step_stats)
end

"""
Führt die Verarbeitung für eine Liste von Polytopen durch.
GEÄNDERT: Nimmt 'display_dim_str' und 'source_description' statt 'dim' und 'config.source_file'.
GEÄNDERT: Parst config.terminal_output und steuert die Ausgabe über Flags.
"""
function run_processing(polytopes::Vector{Matrix{Int}}, display_dim_str::String, source_description::String, config::Config, log_stream)

    # --- NEU: Terminal-Ausgabe-Flags parsen ---
    components_str = lowercase(replace(config.terminal_output, " " => ""))
    if components_str == "true" # Abwärtskompatibilität
        components_str = "all"
    elseif components_str == "false" # Abwärtskompatibilität
        components_str = ""
    end

    show_initial = occursin("initial", components_str) || occursin("all", components_str)
    show_running = occursin("running", components_str) || occursin("all", components_str)
    show_table = occursin("table", components_str) || occursin("all", components_str)
    show_final = occursin("final", components_str) || occursin("all", components_str)
    # --- Ende: Parsing ---

    if show_initial # GEÄNDERT
        # --- Terminal Summary ---
        term_summary_buf = IOBuffer()
        println(term_summary_buf, "Run started at:                      $(Dates.format(now(), "HH:MM:SS"))")
        println(term_summary_buf, "Number of threads:                   $(nthreads())")
        # GEÄNDERT: Verwendet display_dim_str
        println(term_summary_buf, "Detected Dimension:                  $(display_dim_str)") 
        println(term_summary_buf, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
        println(term_summary_buf, "Intersection backend selected:       $(config.intersection_backend)")
        println(term_summary_buf, "Validation enabled:                  $(config.validate)")
        # GEÄNDERT: Verwendet source_description
        println(term_summary_buf, "Input source:                        $(source_description)")
        println(term_summary_buf, "Number of polytopes found:           $(length(polytopes))")
        println(term_summary_buf, "Restricting to unimodular simplices: $(config.only_unimodular)")
        if !isnothing(log_stream)
            # (log_file ist nicht mehr in config, Info wird nicht mehr gedruckt)
        end
        println(term_summary_buf, "")
        print(stdout, String(take!(term_summary_buf)))
    end

    # --- Log File Summary (unverändert, da nicht Terminal) ---
    if !isnothing(log_stream)
        log_summary_buf = IOBuffer()
        println(log_summary_buf, "Number of threads:                   $(nthreads())")
        # GEÄNDERT: Verwendet display_dim_str
        println(log_summary_buf, "Detected Dimension:                  $(display_dim_str)")
        println(log_summary_buf, "Solve mode:                          $(config.find_all ? "Find All" : "Find First")")
        println(log_summary_buf, "Intersection backend selected:       $(config.intersection_backend)")
        println(log_summary_buf, "Validation enabled:                  $(config.validate)")
        # GEÄNDERT: Verwendet source_description
        println(log_summary_buf, "Input source:                        $(source_description)")
        println(log_summary_buf, "Number of polytopes found:           $(length(polytopes))")
        println(log_summary_buf, "Restricting to unimodular simplices: $(config.only_unimodular)")
        println(log_summary_buf, "")
        print(log_stream, String(take!(log_summary_buf)))
        flush(log_stream)
    end

    t_start_run = time()
    total_solutions_found = 0
    triangulations_found_count = 0
    non_triangulatable_count = 0
    recent_times = Float64[]
    is_first_single_line_update = true
    results = ProcessResult[]

    for (i, P) in enumerate(polytopes)
        # 'P' (Matrix{Int}) wird übergeben, process_polytope bestimmt die dim selbst
        # GEÄNDERT: Übergibt show_running Flag an process_polytope
        result = process_polytope(P, i, length(polytopes), config, show_running)
        push!(results, result)
        
        if result.num_solutions_found > 0
            triangulations_found_count += 1
        else
            non_triangulatable_count += 1
        end
        total_solutions_found += result.num_solutions_found
        
        push!(recent_times, result.total_time)
        if length(recent_times) > 100
            popfirst!(recent_times)
        end

        if show_running # GEÄNDERT
            if !is_first_single_line_update
                print(stdout, "\u001b[4A") # Bewegt sich 4 Zeilen nach oben, um Block zu überschreiben
            end
            is_first_single_line_update = false
            
            elapsed_time = time() - t_start_run
            
            eta_str = ""
            if !isempty(recent_times)
                avg_time = sum(recent_times) / length(recent_times)
                remaining = length(polytopes) - i
                eta_seconds = avg_time * remaining
                eta_str = format_duration(eta_seconds)
            end

            # 4-Zeilen-Block
            @printf(stdout, "\r%-40s %s\u001b[K\n", "Elapsed Time:", format_duration(elapsed_time))
            @printf(stdout, "\r%-40s %s\u001b[K\n", "Estimated Time Left:", eta_str)
            @printf(stdout, "\r%-40s \u001b[32m%d\u001b[0m\u001b[K\n", "Triangulatable:", triangulations_found_count)
            @printf(stdout, "\r%-40s \u001b[31m%d\u001b[0m\u001b[K\n", "Non-Triangulatable:", non_triangulatable_count)
            # 1-Zeilen-Status
            print(stdout, "\r" * result.minimal_log * "\u001b[K")
            flush(stdout)
        end

        if !isnothing(log_stream)
            print(log_stream, result.verbose_log)
            flush(log_stream)
        end
    end
    
    # --- NEU: Cursor-Steuerung ---
    if show_running
        # Bewegt den Cursor 5 Zeilen nach oben (4-Zeilen-Block + 1 minimal_log Zeile)
        # und löscht den Bildschirm von dort nach unten, um Platz für die
        # finale Zusammenfassung zu machen.
        print(stdout, "\u001b[5A") # 5 Zeilen hoch
        print(stdout, "\u001b[0J") # Von Cursor bis Ende löschen
    end
    # Das alte `println()` [source 64] wurde entfernt.

    total_time_run = time() - t_start_run
    
    avg_solutions_str = ""
    if config.find_all
        avg_solutions_str = @sprintf("Average Solutions/Polytope:      %.2f\n", total_solutions_found / length(polytopes))
    end

    # --- Aggregierte Statistik-Tabelle (wird nur erstellt, wenn show_table true ist) ---
    stats_table_str = ""
    if show_table
        stats_table_buf = IOBuffer()
        if !isempty(results)
            step_times = Dict{String, Vector{Float64}}()
            step_bytes = Dict{String, Vector{Int64}}()
            step_order = String[]

            for res in results
                for stat in res.step_stats
                    if !haskey(step_times, stat.name)
                        step_times[stat.name] = Float64[]
                        step_bytes[stat.name] = Int64[]
                        push!(step_order, stat.name)
                    end
                    push!(step_times[stat.name], stat.duration_s)
                    push!(step_bytes[stat.name], stat.alloc_bytes)
                end
            end

            println(stats_table_buf, "\n--- Detailed Step Statistics (Aggregated) ---")
            println(stats_table_buf, @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s",
                                            "Step Name", "Total Time", "Avg Time", "Max Memory", "Avg Memory"))
            println(stats_table_buf, "-"^89)

            for step_name in step_order
                times = step_times[step_name]
                bytes = step_bytes[step_name]
                
                if isempty(times); continue; end 
                
                total_time = sum(times)
                avg_time = total_time / length(times)
                max_mem = isempty(bytes) ? 0 : maximum(bytes)
                avg_mem = isempty(bytes) ? 0.0 : sum(bytes) / length(bytes)

                println(stats_table_buf, @sprintf("%-35s | %-12s | %-12s | %-12s | %-12s",
                                                step_name,
                                                format_duration(total_time),
                                                @sprintf("%.3f s", avg_time),
                                                format_bytes(max_mem),
                                                format_bytes(avg_mem)))
            end
        end
        stats_table_str = String(take!(stats_table_buf))
    end
    # --- Ende: Aggregierte Statistik-Tabelle ---

    # --- Finale Zusammenfassung (Kern) ---
    summary_core_str = """

    Run finished: $(Dates.format(now(), "HH:MM:SS"))

    ----------------------------------------
    Run Summary
    ----------------------------------------
    Total Polytopes Processed:     $(length(polytopes))
    Triangulatable:                \u001b[32m$triangulations_found_count\u001b[0m
    Non-Triangulatable:            \u001b[31m$non_triangulatable_count\u001b[0m
    $(avg_solutions_str)Total Run Time:                $(format_duration(total_time_run))
    ----------------------------------------
    """

    # --- Getrenntes Drucken der finalen Teile ---
    if show_final
        print(stdout, summary_core_str)
    end
    
    if show_table
        print(stdout, stats_table_str) # Druckt die Tabelle (oder einen leeren String)
        println(stdout) # Fügt einen Zeilenumbruch nach der Tabelle hinzu
    end
    
    # Log-Stream erhält die kombinierte, nicht-farbige Ausgabe
    if !isnothing(log_stream)
        final_log_str = summary_core_str * stats_table_str
        print(log_stream, replace(final_log_str, r"\u001b\[\d+m" => ""))
    end

    return results
end

# --- Öffentliche API-Funktionen ---

"""
NEU: Trianguliert ein einzelnes Polyhedron-Objekt.
GEÄNDERT: terminal_output ist String
"""
function triangulate(polytope::Polyhedron; intersection_backend::String="cpu", only_unimodular::Bool=true, find_all::Bool=false, log_file::String="", terminal_output::String="", validate::Bool=false, plotter::String="")
    vmatrix = _convert_polyhedron_to_vmatrix(polytope)
    if isempty(vmatrix)
        @error("Konnte Polyeder nicht verarbeiten.")
        return nothing
    end
    
    # "Single Polyhedron Input" als Quellbeschreibung
    results = _triangulate([vmatrix], "Single Polyhedron Input", intersection_backend, only_unimodular, find_all, log_file, terminal_output, validate, plotter)
    
    # KORREKTUR: [1] statt [0]
    return isempty(results) ? nothing : results[1] 
end

"""
NEU: Trianguliert einen Vektor von Polyhedron-Objekten.
GEÄNDERT: terminal_output ist String
"""
function triangulate(polytopes::Vector{Polyhedron}; intersection_backend::String="cpu", only_unimodular::Bool=true, find_all::Bool=false, log_file::String="", terminal_output::String="", validate::Bool=false, plotter::String="")
    vmatrices = Matrix{Int}[]
    for p in polytopes
        vmatrix = _convert_polyhedron_to_vmatrix(p)
        if !isempty(vmatrix)
            push!(vmatrices, vmatrix)
        else
            @warn("Überspringe ein Polyeder, das nicht konvertiert werden konnte.")
        end
    end
    
    if isempty(vmatrices)
        @error("Keine Polyeder konnten verarbeitet werden.")
        return ProcessResult[] # Leere Ergebnisliste
    end

    # "Polyhedron List Input" als Quellbeschreibung
    return _triangulate(vmatrices, "Polyhedron List Input", intersection_backend, only_unimodular, find_all, log_file, terminal_output, validate, plotter)
end

"""
Haupt-API-Funktion: Liest Polytope aus einer Datei und startet die Triangulierung.
GEÄNDERT: Übergibt 'path_to_polytopes' als 'source_description' an _triangulate.
GEÄNDERT: terminal_output ist String
"""
function triangulate(path_to_polytopes::String; intersection_backend::String="cpu", only_unimodular::Bool=true, find_all::Bool=false, log_file::String="", terminal_output::String="", validate::Bool=false, plotter::String="")
    local polytopes
    try
        polytopes = read_polytopes_from_file(path_to_polytopes)
        if isempty(polytopes); @error("Error: No polytopes loaded from '$path_to_polytopes'."); return ProcessResult[]; end
    catch e
        @error("Error loading polytopes from '$path_to_polytopes': '$e'")
        return ProcessResult[]
    end
    # Übergibt den Pfad als Quellbeschreibung
    return _triangulate(polytopes, path_to_polytopes, intersection_backend, only_unimodular, find_all, log_file, terminal_output, validate, plotter)
end

"""
Interne Hauptfunktion, die den Lauf steuert.
GEÄNDERT:
- Nimmt 'source_description' statt 'source_file'.
- Entfernt 'source_file' aus der Config-Erstellung.
- Ermittelt 'display_dim_str' (kann "Mixed" sein) statt 'dim'.
- Übergibt 'display_dim_str' und 'source_description' an 'run_processing'.
- GEÄNDERT: terminal_output ist String
"""
function _triangulate(polytopes::Vector{Matrix{Int}}, source_description::String, intersection_backend::String="cpu", only_unimodular::Bool=true, find_all::Bool=false, log_file::String="", terminal_output::String="", validate::Bool=false, plotter::String="")

    # GEÄNDERT: Config-Erstellung ohne source_file, terminal_output ist String
    config = Config(terminal_output, only_unimodular, intersection_backend, find_all, validate, plotter)

    # --- Pre-run Checks ---
    if config.intersection_backend =="gpu"
        if !CUDA_PACKAGES_LOADED[]; @warn "GPU backend requested, but CUDA not loaded. Falling back to CPU."; config.intersection_backend = "cpu";
        elseif !CUDA.functional(); @warn "CUDA loaded, but no functional GPU found. Falling back to CPU."; config.intersection_backend = "cpu"; end
    end

    log_stream = nothing
    results = ProcessResult[] 
    
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

        # --- GEÄNDERT: Dimension Detection (erlaubt gemischte Dimensionen) ---
        local display_dim_str = "N/A"
        if isempty(polytopes)
            @warn("No polytopes provided to _triangulate.")
            return results
        else
            first_dim = size(polytopes[1], 2)
            # Prüfen, ob alle Polytope die gleiche Dimension haben
            all_same_dim = all(size(p, 2) == first_dim for p in polytopes)
            display_dim_str = all_same_dim ? string(first_dim) : "Mixed"
        end
        
        # GEÄNDERT: Übergibt display_dim_str und source_description
        results = run_processing(polytopes, display_dim_str, source_description, config, log_stream)
        
    finally
        !isnothing(log_stream) && close(log_stream)
    end
    
    return results
end

# if abspath(PROGRAM_FILE) == @__FILE__
#     main() # TODO: patchwork fix; don't run main when importing
# end

end # Ende Modul Triangulate

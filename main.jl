# unimodularTriangulationSAT.jl - v3.2 (mit Validierung)
# Finds unimodular triangulations of 3D, 4D, 5D, and 6D lattice polytopes.

using Oscar: convex_hull, lattice_points
using Combinatorics
using LinearAlgebra
using Polyhedra
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

# Include GPU backends for each dimension
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

include("Intersection_backends/cpu_intersection_3d.jl")
include("Intersection_backends/cpu_intersection.jl")

const CMS_LOADED = Ref(false)
try
    using CryptoMiniSat
    CMS_LOADED[] = true
catch
end

# --- Data Structures ---
mutable struct Config
    polytopes_file::String
    log_file::String
    process_range::String
    processing_order::String
    sort_by::String
    intersection_backend::String
    find_all_simplices::Bool
    solve_mode::String
    validate::Bool # NEU: Hinzugefügt für die Validierungsoption
    terminal_output::String
    file_output::String
    show_initial_vertices::Bool
    show_solution_simplices::Bool
    show_timing_summary::Bool
    solver::String
    plotter_script::String
    plot_range::String
end

struct ProcessResult
    id::Int
    num_solutions_found::Int
    total_time::Float64
    num_lattice_points::Int
    num_simplices_considered::Int
    validation_status::Symbol # NEU: :not_run, :passed, :failed
    verbose_log::String
    minimal_log::String
    first_solution_simplices::Vector{Matrix{Int}}
end

# --- Utility and Configuration Functions ---

function load_config(filepath::String)
    config_dict = TOML.parsefile(filepath)
    files = get(config_dict, "files", Dict())
    run_settings = get(config_dict, "run_settings", Dict())
    output_levels = get(config_dict, "output_levels", Dict())
    verbose_options = get(config_dict, "verbose_options", Dict())
    solver_options = get(config_dict, "solver_options", Dict())
    plotting = get(config_dict, "plotting", Dict())
    return Config(
        get(files, "polytopes_file", "polytopes.txt"),
        get(files, "log_file", ""),
        get(run_settings, "process_range", "1-"),
        get(run_settings, "processing_order", "normal"),
        get(run_settings, "sort_by", "none"),
        get(run_settings, "intersection_backend", "cpu"),
        get(run_settings, "find_all_simplices", false),
        get(run_settings, "solve_mode", "first"),
        get(run_settings, "validate", false), # NEU: Auslesen der Validierungsoption
        get(output_levels, "terminal_output", "verbose"),
        get(output_levels, "file_output", "verbose"),
        get(verbose_options, "show_initial_vertices", true),
        get(verbose_options, "show_solution_simplices", true),
        get(verbose_options, "show_timing_summary", true),
        get(solver_options, "solver", "PicoSAT"),
        get(plotting, "plotter_script", ""),
        get(plotting, "plot_range", "")
    )
end

function validate_config(config::Config)
    valid_options = Dict(
        "processing_order" => ["normal", "reversed", "random"],
        "sort_by" => ["none", "P", "S"],
        "intersection_backend" => ["cpu", "gpu_rationals", "gpu_floats"],
        "solve_mode" => ["first", "all", "count"],
        "terminal_output" => ["verbose", "multi-line", "single-line", "none"],
        "file_output" => ["verbose", "minimal", "none"],
        "solver" => ["PicoSAT", "CryptoMiniSat"]
    )

    for (field, values) in valid_options
        config_value = getfield(config, Symbol(field))
        if !(config_value in values)
            @warn "Invalid '$field' in config: '$(config_value)'. Expected one of $values."
        end
    end
end


function format_duration(total_seconds::Float64)
    total_seconds_int = floor(Int, total_seconds)
    h = total_seconds_int ÷ 3600
    rem_seconds = total_seconds_int % 3600
    m = rem_seconds ÷ 60
    s = rem_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
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

# for ?
function parse_range(range_str::String, max_index::Int)
    indices = Set{Int}()
    parts = split(range_str)
    for part in parts
        part = strip(part)
        if isempty(part); continue; end
        
        if part == "-"
            union!(indices, 1:max_index)
        elseif endswith(part, "-")
            start = parse(Int, part[1:end-1])
            union!(indices, start:max_index)
        elseif startswith(part, "-")
            stop = parse(Int, part[2:end])
            union!(indices, 1:stop)
        elseif contains(part, "-")
            start_str, stop_str = split(part, "-")
            start = parse(Int, start_str)
            stop = parse(Int, stop_str)
            union!(indices, start:stop)
        else
            idx = parse(Int, part)
            push!(indices, idx)
        end
    end
    return sort(collect(indices))
end

# for logs
function update_line(message::String)
    print(stdout, "\r" * message * "\u001b[K"); flush(stdout)
end


# --- Presolve functions ---
# Functions here are used before passing to the intersection kernels

# Computes the lattice points of a polytope (from its V-representation) via Oscar. Care is taken to convert between Oscar and Julia types
# Calling Oscar and the conversion may be time-inefficient, but this presolve is anyways not a bottleneck
function lattice_points_via_Oscar(vertices::Matrix{Int})
    # Build convex hull polytope
    polytope = convex_hull(vertices)
    
    # Get lattice points
    LP = lattice_points(polytope) # This returns a weird Oscar object: "SubObjectIterator{PointVector{ZZRingElem}}"
    
    # Retrieve dimension
    dims = size(LP)

    # Rows
    nrows = dims[1] 
    # Columns
    ncols = size(LP[1])[1] # The dimensions of "SubObjectIterator{PointVector{ZZRingElem}}" are given weirdly

    julia_matrix_LP = [Int(LP[i][j]) for i in 1:nrows, j in 1:ncols] # Conversion from Oscar ZZRingElem type to Julia Int64 type
    return julia_matrix_LP
end


"""
    all_simplices(lattice_points::Matrix{Int}; unimodular_only::Bool=false)

Return all index tuples of (d+1)-element subsets of points in `P` that form
non-degenerate simplices in ℚᵈ, optionally restricting to unimodular simplices
(|det| == 1).

TODO: make this work with arbitrary type ?
"""
function all_simplices(lattice_points::Matrix{Int}; unimodular_only::Bool=false)
    n, d = size(lattice_points)
    simplex_indices = Vector{NTuple{d+1, Int}}()
    if n < d + 1
        return simplex_indices
    end

    for inds in combinations(1:n, d + 1)
        p0 = lattice_points[inds[1], :]
        # build matrix of difference vectors from p0
        M = vcat([(lattice_points[inds[i], :] - p0)' for i in 2:(d + 1)]...)
        # compute the determinant
        det_val = det(M)
        if det_val != 0 && (!unimodular_only || abs(det_val) == 1)
            # if unimodular, add it
            push!(simplex_indices, Tuple(inds))
        end
    end
    return simplex_indices
end

# Computes internal faces of simplices.
# TODO: make this not require floats? The Polyhedra pacakge seems to always use floats...
function precompute_internal_faces(vertices::Matrix{Int}, dim::Int)
    n = size(vertices, 1)
    if n < dim
        return Set{NTuple{dim, Int}}()
    end

    # Build polyhedron and collect halfspaces
    poly = Polyhedra.polyhedron(vrep(vertices))
    hr = hrep(poly)
    planes = collect(halfspaces(hr))

    # Generate all possible faces (index combinations)
    potential_faces = collect(combinations(1:n, dim))

    # Shared atomic counter to distribute work
    next_idx = Threads.Atomic{Int}(1)

    # Spawn one task per thread
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

                # Check whether face lies on boundary plane
                on_boundary = any(plane -> all(iszero, face_points * plane.a .- plane.β), planes)

                if !on_boundary
                    push!(local_faces, Tuple(sort(collect(face_indices))))
                end
            end
            local_faces
        end
        for _ in 1:nthreads()
    ]

    # Merge thread results
    return union(fetch.(tasks)...)
end

# --- Main Processing Functions ---
# File reading and processing before passing to intersection kernel

function process_polytope(initial_vertices::Matrix{Int}, id::Int, run_idx::Int, total_in_run::Int, config::Config)
    dim = size(initial_vertices, 2)
    buf = IOBuffer()
    timings = Vector{Pair{String, Float64}}()
    t_start_total = time_ns()
    #initial_vertices = Rational{BigInt}.(initial_vertices_int)
    validation_status = :not_run # NEU: Initialisierung des Validierungsstatus

    function log_verbose(msg...; is_display::Bool=false)
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = if is_display; sprint(show, "text/plain", msg[1]); else; join(msg, " "); end
        full_msg = "[$timestamp] " * s_msg
        println(buf, full_msg)
        if config.terminal_output == "verbose"; println(stdout, full_msg); flush(stdout); end
    end
    
    log_verbose("Processing $(dim)D Polytope #$id")
    if config.show_initial_vertices; log_verbose("Initial vertices provided:"); log_verbose(initial_vertices, is_display=true); end

    log_verbose("Step 1: Finding all lattice points..."); t_start = time_ns()
    
    # Retrieve the lattice points of P
    P = lattice_points_via_Oscar(initial_vertices)
    
    # -----Logs update-----
    push!(timings, "Find all lattice points" => (time_ns() - t_start) / 1e9); num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if config.terminal_output in ["multi-line", "single-line"]; update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points..."); end
    simplex_search_type = config.find_all_simplices ? "non-degenerate" : "unimodular"
    log_verbose("Step 2: Searching for $simplex_search_type $(dim)-simplices..."); t_start = time_ns()
    # ---------------------
    
    # Retrieve the simplices of P
    S_indices = all_simplices(P, unimodular_only=!config.find_all_simplices)

    # -----Logs update-----
    push!(timings, "Find all $simplex_search_type simplices" => (time_ns() - t_start) / 1e9); num_simplices_found = length(S_indices)
    log_verbose("-> Found $num_simplices_found simplices. Step 2 complete.\n")
    if config.terminal_output in ["multi-line", "single-line"]; update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices_found..."); end
    # Check if we found *no* (unimodular) simplices
    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        msg = "no $simplex_search_type simplices exist"
        minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> \u001b[31m%s\u001b[0m", run_idx, total_in_run, num_lattice_points, num_simplices_found, msg)
        verbose_log = String(take!(buf)) * "\nNo simplices available. Cannot proceed."
        if config.terminal_output == "verbose"
            println(stdout, verbose_log)
        end
        return ProcessResult(id, 0, total_time, num_lattice_points, num_simplices_found, :not_run, verbose_log, minimal_log, [])
    end
    log_verbose("Step 3: Precomputing internal faces..."); t_start = time_ns()
    # ---------------------

    # --- Precomputing internal faces (dimension-agnostic function) ---

    internal_faces_set = precompute_internal_faces(P, dim)

    push!(timings, "Precompute internal faces (parallel)" => (time_ns() - t_start) / 1e9)
    log_verbose("-> Found $(length(internal_faces_set)) unique internal faces. Step 3 complete.\n")

    num_simplices = length(S_indices); cnf = Vector{Vector{Int}}(); push!(cnf, collect(1:num_simplices))

    log_verbose("Step 4a: Generating intersection clauses..."); t_start = time_ns()
    
    intersection_clauses = let n_simplices = num_simplices
        intersect_func = nothing
        use_gpu = false

        if config.intersection_backend == "gpu_rationals"
            if dim == 3 && isdefined(Main, :GPUIntersection3D)
                log_verbose("     Using 3D GPU backend (Rationals)...")
                intersect_func = () -> Main.GPUIntersection3D.get_intersecting_pairs_gpu(P, S_indices)
                use_gpu = true
            elseif dim == 4 && isdefined(Main, :GPUIntersection4D)
                log_verbose("     Using 4D GPU backend (Rationals)...")
                intersect_func = () -> Main.GPUIntersection4D.get_intersecting_pairs_gpu_4d(P, S_indices)
                use_gpu = true
            elseif dim == 5 && isdefined(Main, :GPUIntersection5D)
                log_verbose("     Using 5D GPU backend (Rationals)...")
                intersect_func = () -> Main.GPUIntersection5D.get_intersecting_pairs_gpu_5d(P, S_indices)
                use_gpu = true
            elseif dim == 6 && isdefined(Main, :GPUIntersection6D)
                log_verbose("     Using 6D GPU backend (Rationals)...")
                intersect_func = () -> Main.GPUIntersection6D.get_intersecting_pairs_gpu_6d(P, S_indices)
                use_gpu = true
            end
        elseif config.intersection_backend == "gpu_floats"
            if dim == 3 && isdefined(Main, :GPUIntersection3DFloats)
                log_verbose("     Using 3D GPU backend (Floats)...")
                intersect_func = () -> Main.GPUIntersection3DFloats.get_intersecting_pairs_gpu(P, S_indices)
                use_gpu = true
            elseif dim == 4 && isdefined(Main, :GPUIntersection4DFloats)
                log_verbose("     Using 4D GPU backend (Floats)...")
                intersect_func = () -> Main.GPUIntersection4DFloats.get_intersecting_pairs_gpu_4d(P, S_indices)
                use_gpu = true
            elseif dim == 5 && isdefined(Main, :GPUIntersection5DFloats)
                log_verbose("     Using 5D GPU backend (Floats)...")
                intersect_func = () -> Main.GPUIntersection5DFloats.get_intersecting_pairs_gpu_5d(P, S_indices)
                use_gpu = true
            elseif dim == 6 && isdefined(Main, :GPUIntersection6DFloats)
                log_verbose("     Using 6D GPU backend (Floats)...")
                intersect_func = () -> Main.GPUIntersection6DFloats.get_intersecting_pairs_gpu_6d(P, S_indices)
                use_gpu = true
            end
        end
        
        if use_gpu && !isnothing(intersect_func)
            intersect_func() # Execute the selected GPU function
        else
            if startswith(config.intersection_backend, "gpu")
                 log_verbose("     WARNING: GPU backend '$(config.intersection_backend)' for $(dim)D not available. Falling back to CPU.")
            end
            log_verbose("     Using CPU backend.")
            CPUIntersection.get_intersecting_pairs_cpu_generic(P, S_indices)
        end
    end

    if config.terminal_output == "verbose"; update_line(""); println(stdout); end
    push!(timings, "Generate intersection clauses" => (time_ns() - t_start) / 1e9); 
    append!(cnf, intersection_clauses); log_verbose("-> Found $(length(intersection_clauses)) intersection clauses. Step 4a complete.\n")

    log_verbose("Step 4b: Generating face-covering clauses..."); t_start = time_ns(); face_dim = dim
    face_clauses = let n_simplices = num_simplices
    next_simplex_idx = Threads.Atomic{Int}(1)

        # one future per thread
        tasks = [
            Threads.@spawn begin
                local_clauses = Vector{Vector{Int}}()
                while true
                    i = Threads.atomic_add!(next_simplex_idx, 1)
                    if i > n_simplices
                        break
                    end

                    if config.terminal_output == "verbose" && threadid() == 1 &&
                        (i % 100 == 0 || i == n_simplices)
                        update_line("[$(Dates.format(now(), "HH:MM:SS"))] ... checking for face covers: $i / $n_simplices")
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

        # combine results
        vcat(fetch.(tasks)...)
    end
    if config.terminal_output == "verbose"; update_line(""); println(stdout); end
    append!(cnf, face_clauses)
    push!(timings, "Generate face-covering clauses" => (time_ns() - t_start) / 1e9)
    log_verbose("-> Found $(length(face_clauses)) face-covering clauses. Step 4b complete.\n")
    
    log_verbose("Step 5: Handing SAT problem to solver ($(config.solver))...");
    log_verbose("     Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")
    log_verbose("     This step may take a long time if the problem is complex.")
    if config.terminal_output in ["multi-line", "single-line"]; update_line("($(@sprintf("%d / %d", run_idx, total_in_run))): |P|=$num_lattice_points |S|=$num_simplices_found solving..."); end
    
    t_start_solve = time_ns(); solutions = []; num_solutions = 0
    solver_used = config.solver
    if solver_used == "CryptoMiniSat" && !CMS_LOADED[]; log_verbose("Warning: CryptoMiniSat not available, falling back to PicoSAT."); solver_used = "PicoSAT"; end

    solver_func = solver_used == "CryptoMiniSat" ? CryptoMiniSat : PicoSAT
    if config.solve_mode == "first"
        solution = solver_func.solve(cnf); if solution isa Vector{Int}; num_solutions = 1; push!(solutions, solution); end
    else
        for solution in solver_func.itersolve(cnf)
            num_solutions += 1; if config.solve_mode == "all"; push!(solutions, solution); end;
        end
    end

    log_verbose("-> SAT solver finished. Step 5 complete.")
    push!(timings, "Solve SAT problem" => (time_ns() - t_start_solve) / 1e9);
    
    # --- NEUER VALIDIERUNGSBLOCK ---
    first_sol_indices = Int[]
    if num_solutions > 0
        first_sol_indices = findall(l -> l > 0, first(solutions))
    end

    if config.validate && num_solutions > 0 && config.intersection_backend != "cpu"
        log_verbose("\nStep 5a: Validating solution using precise CPU intersection checker...")
        validation_status = :passed # Annahme: bestanden, bis Fehler gefunden wird
        cpu_intersect_func = CPUIntersection.simplices_intersect_sat_cpu
        solution_simplices_to_check = collect(first_sol_indices)
        num_check = length(solution_simplices_to_check)
        intersection_found_by_cpu = false

        for i in 1:num_check
            if intersection_found_by_cpu; break; end
            s1_point_indices = S_indices[solution_simplices_to_check[i]]
            s1_points = P[collect(s1_point_indices), :]

            for j in (i + 1):num_check
                s2_point_indices = S_indices[solution_simplices_to_check[j]]
                s2_points = P[collect(s2_point_indices), :]
                
                if cpu_intersect_func(s1_points, s2_points)
                    log_verbose("   VALIDATION FAILED: Intersection found between solution simplex $i and $j by CPU checker.")
                    log_verbose("   Simplex $i indices: $s1_point_indices")
                    log_verbose("   Simplex $j indices: $s2_point_indices")
                    validation_status = :failed
                    intersection_found_by_cpu = true
                    break
                end
            end
        end
        
        if validation_status == :passed
            log_verbose("   VALIDATION SUCCESSFUL: No intersections found among solution simplices by CPU checker.")
        else
            num_solutions = 0 # Markiere als ungültig, wenn Validierung fehlschlägt
        end
        log_verbose("-> Validation complete. Step 5a complete.")
    elseif num_solutions > 0 && config.validate && config.intersection_backend == "cpu"
        log_verbose("\nStep 5a: Validation skipped (CPU backend already performed exact checks).")
        validation_status = :passed
    end
    # --- ENDE VALIDIERUNGSBLOCK ---

    total_time = (time_ns() - t_start_total) / 1e9; push!(timings, "Total execution time" => total_time)
    log_verbose("\n$(num_solutions) valid triangulation(s) found.")
    
    first_solution_simplices = Vector{Matrix{Int}}()
    if num_solutions > 0 && !isempty(first_sol_indices) # first_sol_indices wurde bereits oben berechnet
        first_solution_simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in first_sol_indices]
    end

    if config.show_solution_simplices && !isempty(first_solution_simplices)
        log_verbose("\nDisplaying first valid triangulation:");
        for s in first_solution_simplices; log_verbose(s, is_display=true); end
    end
    
    if num_solutions > 0 && !isempty(config.plot_range) && !isempty(config.plotter_script) && (id in parse_range(config.plot_range, 1_000_000))
        log_verbose("\nStep 6: Plotting results...")
        if dim == 3
            temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(first_solution_simplices)); close(temp_io); run(`python $(config.plotter_script) $(temp_path)`); finally rm(temp_path, force=true); end
        elseif dim == 4
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices)); boundary_planes = collect(halfspaces(hrep(initial_poly)))
            solution_simplices_rational = [P[collect(S_indices[i]), :] for i in findall(l -> l > 0, first(solutions))]
            for (plane_idx, plane) in enumerate(boundary_planes)
                facet_triangulation_4D = [s for s in solution_simplices_rational if count(v -> iszero(dot(plane.a, v) - plane.β), eachrow(s)) == 4]
                if isempty(facet_triangulation_4D); continue; end
                log_verbose("     Plotting induced 3D triangulation for facet #$plane_idx...")
                origin_4d = facet_triangulation_4D[1][1,:]; basis_3d = get_orthonormal_basis(plane.a)
                projected_simplices = Vector{Matrix{Int}}()
                for s in facet_triangulation_4D
                    face_vertices_on_plane = filter(v -> iszero(dot(plane.a, v) - plane.β), eachrow(s))
                    if length(face_vertices_on_plane) == 4
                        projected_verts_3d = [round.(Int, [dot(v - origin_4d, b) for b in basis_3d]) for v in face_vertices_on_plane]
                        push!(projected_simplices, vcat(projected_verts_3d'...))
                    end
                end
                temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(projected_simplices)); close(temp_io); run(`python $(config.plotter_script) $(temp_path)`); finally rm(temp_path, force=true); end
            end
        elseif dim == 5
            log_verbose("     Plotting induced 3D triangulations for 3-faces...")
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices))
            boundary_planes = collect(halfspaces(hrep(initial_poly)))
            solution_simplices_rational = [P[collect(S_indices[i]), :] for i in findall(l -> l > 0, first(solutions))]

            for i in 1:length(boundary_planes)
                for j in (i + 1):length(boundary_planes)
                    plane1 = boundary_planes[i]
                    plane2 = boundary_planes[j]

                    face_simplices_5D = [s for s in solution_simplices_rational if count(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(s)) >= 4]
                    if isempty(face_simplices_5D); continue; end

                    log_verbose("     Plotting 3-face defined by facets #$i and #$j...")
                    
                    origin_5d = first(filter(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(face_simplices_5D[1])))
                    basis_3d = get_orthonormal_basis_for_subspace_3d(plane1.a, plane2.a)
                    
                    projected_simplices = Vector{Matrix{Int}}()
                    for s in face_simplices_5D
                        verts_on_face = filter(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(s))
                        for tetra_verts in combinations(verts_on_face, 4)
                            projected_verts_3d = [round.(Int, [dot(v - origin_5d, b) for b in basis_3d]) for v in tetra_verts]
                            push!(projected_simplices, vcat(projected_verts_3d'...))
                        end
                    end
                    
                    if !isempty(projected_simplices)
                        unique_simplices = unique(s -> Tuple(sortslices(s, dims=1)), projected_simplices)
                        temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(unique_simplices)); close(temp_io); run(`python $(config.plotter_script) $(temp_path)`); finally rm(temp_path, force=true); end
                    end
                end
            end
        elseif dim == 6
            log_verbose("     Plotting induced 3D triangulations for 3-faces...")
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices))
            boundary_planes = collect(halfspaces(hrep(initial_poly)))
            solution_simplices_rational = [P[collect(S_indices[i]), :] for i in findall(l -> l > 0, first(solutions))]

            # Iterate over all triples of facets to find 3-dimensional faces
            for i in 1:length(boundary_planes), j in (i+1):length(boundary_planes), k in (j+1):length(boundary_planes)
                p1, p2, p3 = boundary_planes[i], boundary_planes[j], boundary_planes[k]
                
                face_simplices_6D = [s for s in solution_simplices_rational if count(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(s)) >= 4]
                if isempty(face_simplices_6D); continue; end

                log_verbose("     Plotting 3-face defined by facets #$i, #$j, and #$k...")
                
                origin_6d = first(filter(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(face_simplices_6D[1])))
                basis_3d = get_orthonormal_basis_for_subspace_3d_from_6d(p1.a, p2.a, p3.a)
                
                projected_simplices = Vector{Matrix{Int}}()
                for s in face_simplices_6D
                    verts_on_face = filter(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(s))
                    for tetra_verts in combinations(verts_on_face, 4)
                        projected_verts_3d = [round.(Int, [dot(v - origin_6d, b) for b in basis_3d]) for v in tetra_verts]
                        push!(projected_simplices, vcat(projected_verts_3d'...))
                    end
                end
                
                if !isempty(projected_simplices)
                    unique_simplices = unique(s -> Tuple(sortslices(s, dims=1)), projected_simplices)
                    temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(unique_simplices)); close(temp_io); run(`python $(config.plotter_script) $(temp_path)`); finally rm(temp_path, force=true); end
                end
            end
        end
        log_verbose("-> Plotting complete. Step 6 complete.")
    end

    if config.show_timing_summary
        summary_buf = IOBuffer()
        println(summary_buf, "\n--- Timing & Memory Summary for Polytope #$id ---")
        peak_ram_bytes = Sys.maxrss(); for (op, dur) in timings; println(summary_buf, @sprintf("%-45s: %.4f seconds", op, dur)); end
        println(summary_buf, @sprintf("%-45s: %.2f MiB", "Peak memory usage (Max RSS)", peak_ram_bytes / 1024^2))
        log_verbose(String(take!(summary_buf)))
    end
    
    # NEU: Angepasste Ergebniserstellung basierend auf Validierung
    result_str = ""
    if validation_status == :failed
        result_str = @sprintf("\u001b[31mVALIDATION FAILED\u001b[0m after %.2f s", total_time)
    elseif num_solutions > 0
        result_str = @sprintf("\u001b[32mfound %d solution(s)\u001b[0m in %.2f s", num_solutions, total_time)
    else
        result_str = @sprintf("\u001b[31mno solution exists\u001b[0m, searched for %.2f s", total_time)
    end
    minimal_log = @sprintf("(%d / %d): |P|=%d |S|=%d -> %s", run_idx, total_in_run, num_lattice_points, num_simplices_found, result_str)
    
    return ProcessResult(id, num_solutions, total_time, num_lattice_points, num_simplices_found, validation_status, String(take!(buf)), minimal_log, first_solution_simplices)
end

function run_processing(polytopes::Vector{Matrix{Int}}, dim::Int, config::Config, range_to_process, log_stream)
    indices_to_process = collect(range_to_process)

    if config.sort_by in ["#Lattice Points", "#Simplices"]
        println("Pre-calculating metrics for sorting by '$(config.sort_by)'...")
        metrics = []
        total = length(indices_to_process)
        for (i, p_idx) in enumerate(indices_to_process)
            update_line("Sorting pre-calculation: $i / $total")
            initial_vertices = polytopes[p_idx]
            
            # Retrieve the lattice points of the polytope
            P = lattice_points_via_Oscar(initial_vertices)

            num_points = size(P, 1)

            if config.sort_by == "P"
                push!(metrics, (p_idx, num_points))
            elseif config.sort_by == "S"
                S_indices = all_simplices(P, unimodular_only=!config.find_all_simplices)
                num_simplices = length(S_indices)
                push!(metrics, (p_idx, num_simplices))
            end
        end
        println()
        
        sort!(metrics, by = x -> x[2])
        indices_to_process = [m[1] for m in metrics]
    end

    if config.processing_order == "reversed"; reverse!(indices_to_process); elseif config.processing_order == "random"; shuffle!(indices_to_process); end

    if config.terminal_output != "none"
        # --- Terminal Summary ---
        term_summary_buf = IOBuffer()
        println(term_summary_buf, "Run started at:                          $(Dates.format(now(), "HH:MM:SS"))")
        println(term_summary_buf, "Number of threads:                       $(nthreads())")
        println(term_summary_buf, "Detected Dimension:                      $(dim)")
        println(term_summary_buf, "Solver requested:                        $(config.solver)")
        println(term_summary_buf, "Solve mode:                              $(config.solve_mode)")
        println(term_summary_buf, "Intersection backend selected:           $(config.intersection_backend)")
        println(term_summary_buf, "Validation enabled:                      $(config.validate)") # NEU
        println(term_summary_buf, "Input file:                              $(config.polytopes_file)")
        println(term_summary_buf, "Number of polytopes found:               $(length(polytopes))")
        println(term_summary_buf, "Processing range:                        $(config.process_range)")
        println(term_summary_buf, "Number of polytopes to process:          $(length(indices_to_process))")
        println(term_summary_buf, "Restricting to unimodular simplices:     $(!config.find_all_simplices)")
        if !isnothing(log_stream)
            println(term_summary_buf, "Writing to log file:                     $(config.log_file)")
        end
        println(term_summary_buf, "")
        print(stdout, String(take!(term_summary_buf)))

        # --- Log File Summary ---
        if !isnothing(log_stream)
            log_summary_buf = IOBuffer()
            println(log_summary_buf, "Number of threads:                       $(nthreads())")
            println(log_summary_buf, "Detected Dimension:                      $(dim)")
            println(log_summary_buf, "Solver requested:                        $(config.solver)")
            println(log_summary_buf, "Solve mode:                              $(config.solve_mode)")
            println(log_summary_buf, "Intersection backend selected:           $(config.intersection_backend)")
            println(log_summary_buf, "Validation enabled:                      $(config.validate)") # NEU
            println(log_summary_buf, "Input file:                              $(config.polytopes_file)")
            println(log_summary_buf, "Number of polytopes found:               $(length(polytopes))")
            println(log_summary_buf, "Processing range:                        $(config.process_range)")
            println(log_summary_buf, "Number of polytopes to process:          $(length(indices_to_process))")
            println(log_summary_buf, "Restricting to unimodular simplices:     $(!config.find_all_simplices)")
            println(log_summary_buf, "")
            print(log_stream, String(take!(log_summary_buf)))
            flush(log_stream)
        end
    end

    t_start_run = time(); total_solutions_found = 0; triangulations_found_count = 0; non_triangulatable_count = 0
    recent_times = Float64[] # For ETA calculation
    is_first_single_line_update = true

    for (i, p_idx) in enumerate(indices_to_process)
        result = process_polytope(polytopes[p_idx], p_idx, i, length(indices_to_process), config)
        
        # NEU: Prüfe Validierungsstatus bei der Zählung
        if result.num_solutions_found > 0 && result.validation_status != :failed
            triangulations_found_count += 1
        else
            non_triangulatable_count += 1
        end
        # Nur gültige Lösungen zur Gesamtzahl addieren, wenn die Validierung bestanden wurde oder nicht lief
        if result.validation_status != :failed
            total_solutions_found += result.num_solutions_found
        end
        
        push!(recent_times, result.total_time)
        if length(recent_times) > 100
            popfirst!(recent_times)
        end

        if config.terminal_output == "single-line"
            if !is_first_single_line_update
                print(stdout, "\u001b[4A")
            end
            is_first_single_line_update = false
            
            elapsed_time = time() - t_start_run
            
            eta_str = "Calculating..."
            if !isempty(recent_times)
                avg_time = sum(recent_times) / length(recent_times)
                remaining = length(indices_to_process) - i
                eta_seconds = avg_time * remaining
                eta_str = format_duration(eta_seconds)
            end

            @printf(stdout, "\r%-40s %s\u001b[K\n", "Elapsed Time:", format_duration(elapsed_time))
            @printf(stdout, "\r%-40s %s\u001b[K\n", "Estimated Time Left:", eta_str)
            @printf(stdout, "\r%-40s \u001b[32m%d\u001b[0m\u001b[K\n", "Triangulatable:", triangulations_found_count)
            @printf(stdout, "\r%-40s \u001b[31m%d\u001b[0m\u001b[K\n", "Non-Triangulatable:", non_triangulatable_count)
            print(stdout, "\r" * result.minimal_log * "\u001b[K")
            flush(stdout)
        elseif config.terminal_output == "multi-line"
            println(stdout, result.minimal_log)
        end

        if !isnothing(log_stream)
            log_content = config.file_output == "verbose" ? result.verbose_log : result.minimal_log * "\n"
            print(log_stream, log_content); flush(log_stream)
        end
    end
    
    if config.terminal_output == "single-line"; println(); end
    total_time_run = time() - t_start_run
    
    avg_solutions_str = ""
    if config.solve_mode in ["all", "count"] && !isempty(indices_to_process)
        avg_solutions_str = @sprintf("Average Solutions/Polytope:      %.2f\n", total_solutions_found / length(indices_to_process))
    end

    summary_str = """

    Run finished: $(Dates.format(now(), "HH:MM:SS"))

    ----------------------------------------
    Run Summary
    ----------------------------------------
    Total Polytopes Processed:       $(length(indices_to_process))
    Triangulatable:                  \u001b[32m$triangulations_found_count\u001b[0m
    Non-Triangulatable:              \u001b[31m$non_triangulatable_count\u001b[0m
    $(avg_solutions_str)Total Run Time:                  $(format_duration(total_time_run))
    ----------------------------------------
    """
    print(stdout, summary_str)
    if !isnothing(log_stream); print(log_stream, replace(summary_str, r"\u001b\[\d+m" => "")); end
end

function main()
    config_path = isempty(ARGS) ? "Configs/config.toml" : ARGS[1]

    if !isfile(config_path); println(stderr, "Error: Config file not found at '$config_path'")
        return
    end
    config = load_config(config_path)
    validate_config(config)

    # --- Pre-run Checks for Available Packages ---
    if startswith(config.intersection_backend, "gpu")
        if !CUDA_PACKAGES_LOADED[]; @warn "GPU backend requested, but CUDA not loaded. Falling back to 'cpu'."; config.intersection_backend = "cpu";
        elseif !CUDA.functional(); @warn "CUDA loaded, but no functional GPU found. Falling back to 'cpu'."; config.intersection_backend = "cpu"; end
    end
    if config.solver == "CryptoMiniSat" && !CMS_LOADED[]; @warn "CryptoMiniSat solver requested, but not loaded. Falling back to PicoSAT."; config.solver = "PicoSAT"; end

    log_stream = nothing
    if !isempty(config.log_file)
        try 
            log_stream = open(config.log_file, "a")
            println(log_stream, "\n\n" * "#"^80, "\n# New Run Started at $(now())\n" * "#"^80)
        catch e
            println(stderr, "Error opening log file: $e")
            return
        end
    end

    try
        polytopes = read_polytopes_from_file(config.polytopes_file)
        if isempty(polytopes); println(stderr, "Error: No polytopes loaded from '$(config.polytopes_file)'."); return; end
        
        range_to_process = parse_range(config.process_range, length(polytopes))
        
        if isempty(range_to_process)
            println(stderr, "Warning: The specified 'process_range' is empty or invalid. Nothing to process.")
            return
        end
        
        if !all(i -> 1 <= i <= length(polytopes), range_to_process)
            println(stderr, "Error: 'process_range' contains indices out of bounds (1 to $(length(polytopes))).")
            return
        end

        # --- Dimension Detection ---
        dim = size(polytopes[first(range_to_process)], 2)
        if !(dim in [3, 4, 5, 6])
            println(stderr, "Error: Detected dimension is $dim. Only 3D, 4D, 5D, and 6D are supported."); return;
        end
        
        run_processing(polytopes, dim, config, range_to_process, log_stream)
        
    finally
        !isnothing(log_stream) && close(log_stream)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main() # TODO: patchwork fix; don't run main when importing
end

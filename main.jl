# rationalsFindUnimodularTriangulationSAT.jl - v2.5 Final
# Finds unimodular triangulations of 3D and 4D lattice polytopes.
using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf
using Base.Threads
using TOML
using Random

# --- Conditional Package Inclusion ---
const CUDA_PACKAGES_LOADED = Ref(false)
try
    using CUDA, StaticArrays, CUDA.Adapt
    CUDA_PACKAGES_LOADED[] = true
catch;
end

if CUDA_PACKAGES_LOADED[] && isfile("gpu_intersection.jl")
    include("gpu_intersection.jl")
end

const CMS_LOADED = Ref(false)
try
    using CryptoMiniSat
    CMS_LOADED[] = true
catch;
end

if CUDA_PACKAGES_LOADED[] && isfile("gpu_intersection_4d.jl")
    include("gpu_intersection_4d.jl")
end

# --- Data Structures ---
mutable struct Config
    polytopes_file::String
    log_file::String
    process_range::String
    processing_order::String
    sort_by::String # New field for sorting
    intersection_backend::String
    find_all_simplices::Bool
    terminal_output::String
    terminal_mode::String
    file_output::String
    solution_reporting::String
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
        get(run_settings, "sort_by", "none"), # Load new setting
        get(run_settings, "intersection_backend", "cpu"),
        get(run_settings, "find_all_simplices", false),
        get(output_levels, "terminal_output", "verbose"),
        get(output_levels, "terminal_mode", "multi-line"),
        get(output_levels, "file_output", "verbose"),
        get(output_levels, "solution_reporting", "first"),
        get(verbose_options, "show_initial_vertices", true),
        get(verbose_options, "show_solution_simplices", true),
        get(verbose_options, "show_timing_summary", true),
        get(solver_options, "solver", "PicoSAT"),
        get(plotting, "plotter_script", ""),
        get(plotting, "plot_range", "")
    )
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

function parse_range(s::String, max_index::Int)
    s = strip(s); if s == "-" || isempty(s); return 1:max_index; end
    if endswith(s, "-"); return parse(Int, s[1:end-1]):max_index; end
    if contains(s, "-"); parts = parse.(Int, split(s, "-")); return parts[1]:parts[2]; end
    idx = parse(Int, s); return idx:idx
end

function update_line(message::String)
    print(stdout, "\r" * message * "\u001b[K"); flush(stdout)
end

function format_simplices_for_plotter(simplices::Vector{Matrix{Int}})
    return "[" * join(["[" * join(["[$(join(v, ","))]" for v in eachrow(s)], ",") * "]" for s in simplices], ",") * "]"
end

# --- Geometric Helper Functions ---

function generalized_cross_product_4d(v1::Vector{T}, v2::Vector{T}, v3::Vector{T}) where T
    M = hcat(v1, v2, v3)
    return [ det(M[[2,3,4], :]), -det(M[[1,3,4], :]), det(M[[1,2,4], :]), -det(M[[1,2,3], :]) ]
end

function get_orthonormal_basis(normal::Vector{Rational{BigInt}})
    # This function generates a 3D orthonormal basis for the hyperplane defined by the normal vector.
    # All calculations are done in Float64 for correctness, as orthonormal vectors are not generally rational.
    # The calling function rounds the projection results to Int, so Float64 is the appropriate type here.
    normal_f64 = Float64.(normal)
    # Handle the unlikely case of a zero normal vector.
    if iszero(norm(normal_f64))
        # Return a standard basis.
        return [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0] ]
    end
    normal_f64 ./= norm(normal_f64)

    # Initialize basis vectors to hold Float64.
    basis = [zeros(Float64, 4) for _ in 1:3]

    # Find a standard basis vector 'v' that is not parallel to the normal vector.
    j = argmax(abs.(normal_f64))
    v = zeros(Float64, 4)
    v[mod1(j + 1, 4)] = 1.0

    # Use the robust Gram-Schmidt process to find an orthonormal basis.
    # First basis vector:
    basis[1] = v - dot(v, normal_f64) * normal_f64
    basis[1] ./= norm(basis[1])

    # Second basis vector:
    # Project another standard basis vector 'k' and make it orthogonal to the first two vectors.
    k = zeros(Float64, 4)
    k_idx = mod1(j + 2, 4)
    # Ensure k is not the same as v
    if k_idx == argmax(v)
        k_idx = mod1(j + 3, 4)
    end
    k[k_idx] = 1.0
    
    basis[2] = k - dot(k, normal_f64) * normal_f64 - dot(k, basis[1]) * basis[1]
    basis[2] ./= norm(basis[2])

    # Third basis vector is the generalized cross product of the normal and the first two basis vectors.
    basis[3] = generalized_cross_product_4d(basis[1], basis[2], normal_f64)
    # Normalize to safeguard against floating-point inaccuracies.
    basis[3] ./= norm(basis[3])

    return basis
end

function _normalize_axis(axis::Vector{Rational{BigInt}})
    if all(iszero, axis); return axis; end
    denominators = [v.den for v in axis]; common_mult = lcm(denominators)
    int_axis = [v.num * (common_mult ÷ v.den) for v in axis]
    common_divisor = gcd(int_axis)
    if common_divisor != 0; int_axis .÷= common_divisor; end
    first_nonzero_idx = findfirst(!iszero, int_axis)
    if first_nonzero_idx !== nothing && int_axis[first_nonzero_idx] < 0; int_axis .*= -1; end
    return Rational{BigInt}.(int_axis)
end

_project_cpu(vertices, axis) = (minimum(vertices * axis), maximum(vertices * axis))

# --- 3D Pipeline Functions ---

function findAllLatticePointsInHull_3d(vertices::Matrix{Rational{BigInt}})
    poly = polyhedron(vrep(vertices)); hr = hrep(poly)
    min_coords = floor.(Int, minimum(convert(Matrix{Float64}, vertices), dims=1)); max_coords = ceil.(Int, maximum(convert(Matrix{Float64}, vertices), dims=1))
    lattice_points = Vector{Vector{Rational{BigInt}}}()
    for iz in min_coords[3]:max_coords[3], iy in min_coords[2]:max_coords[2], ix in min_coords[1]:max_coords[1]
        point = Rational{BigInt}.([ix, iy, iz])
        if all(hr.A * point .<= hr.b); push!(lattice_points, point); end
    end
    return isempty(lattice_points) ? Matrix{Rational{BigInt}}(undef, 0, 3) : vcat(lattice_points'...)
end

function all_simplices_in_3d(P::Matrix{Rational{BigInt}}; unimodular_only::Bool)
    n = size(P, 1); simplex_indices = Vector{NTuple{4, Int}}(); if n < 4 return simplex_indices end
    for inds in combinations(1:n, 4)
        p0, p1, p2, p3 = P[inds[1], :], P[inds[2], :], P[inds[3], :], P[inds[4], :]
        M = vcat((p1 - p0)', (p2 - p0)', (p3 - p0)'); d = det(M)
        if d != 0 && (!unimodular_only || abs(d) == 1); push!(simplex_indices, Tuple(inds)); end
    end
    return simplex_indices
end

function precompute_open_faces_3d(P::Matrix{Rational{BigInt}})
    n = size(P, 1); if n < 3 return Set{NTuple{3, Int}}() end
    poly = polyhedron(vrep(P)); hr = hrep(poly); planes = collect(halfspaces(hr))
    potential_faces = collect(combinations(1:n, 3)); thread_sets = [Set{NTuple{3, Int}}() for _ in 1:nthreads()]
    @threads for face_indices in potential_faces
        face_points = P[collect(face_indices), :]
        on_boundary = any(plane -> all(iszero, face_points * plane.a .- plane.β), planes)
        if !on_boundary; push!(thread_sets[threadid()], Tuple(sort(collect(face_indices)))); end
    end
    return union(thread_sets...)
end

function _get_outward_face_normals_3d(vertices)
    normals = Vector{Vector{Rational{BigInt}}}()
    for face_indices in [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
        p0,p1,p2 = vertices[face_indices[1],:], vertices[face_indices[2],:], vertices[face_indices[3],:]
        normal = cross(p1 - p0, p2 - p0)
        p_fourth = vertices[first(setdiff(1:4, face_indices)), :]
        if dot(normal, p_fourth - p0) > 0; normal = -normal; end
        if !all(iszero, normal); push!(normals, normal); end
    end
    return normals
end

function tetrahedra_intersect_sat_cpu_3d(tetra1_verts, tetra2_verts)
    axes = Vector{Vector{Rational{BigInt}}}()
    append!(axes, _get_outward_face_normals_3d(tetra1_verts)); append!(axes, _get_outward_face_normals_3d(tetra2_verts))
    edges1 = [tetra1_verts[j,:] - tetra1_verts[i,:] for (i, j) in combinations(1:4, 2)]
    edges2 = [tetra2_verts[j,:] - tetra2_verts[i,:] for (i, j) in combinations(1:4, 2)]
    for e1 in edges1, e2 in edges2
        axis = cross(e1, e2)
        if !all(iszero, axis); push!(axes, axis); end
    end
    for axis in axes
        min1, max1 = _project_cpu(tetra1_verts, axis); min2, max2 = _project_cpu(tetra2_verts, axis)
        if max1 <= min2 || max2 <= min1; return false; end
    end
    return true
end

# --- 4D Pipeline Functions ---

function findAllLatticePointsInHull_4d(vertices::Matrix{Rational{BigInt}})
    poly = polyhedron(vrep(vertices)); hr = hrep(poly)
    min_coords = floor.(Int, minimum(convert(Matrix{Float64}, vertices), dims=1)); max_coords = ceil.(Int, maximum(convert(Matrix{Float64}, vertices), dims=1))
    lattice_points = Vector{Vector{Rational{BigInt}}}()
    for iw in min_coords[4]:max_coords[4], iz in min_coords[3]:max_coords[3], iy in min_coords[2]:max_coords[2], ix in min_coords[1]:max_coords[1]
        point = Rational{BigInt}.([ix, iy, iz, iw])
        if all(hr.A * point .<= hr.b); push!(lattice_points, point); end
    end
    return isempty(lattice_points) ? Matrix{Rational{BigInt}}(undef, 0, 4) : vcat(lattice_points'...)
end

function all_simplices_in_4d(P::Matrix{Rational{BigInt}}; unimodular_only::Bool)
    n = size(P, 1); simplex_indices = Vector{NTuple{5, Int}}(); if n < 5 return simplex_indices end
    for inds in combinations(1:n, 5)
        p0 = P[inds[1], :]; M = vcat([(P[inds[i], :] - p0)' for i in 2:5]...); d = det(M)
        if d != 0 && (!unimodular_only || abs(d) == 1); push!(simplex_indices, Tuple(inds)); end
    end
    return simplex_indices
end

function precompute_open_faces_4d(P::Matrix{Rational{BigInt}})
    n = size(P, 1); if n < 4 return Set{NTuple{4, Int}}() end
    poly = polyhedron(vrep(P)); hr = hrep(poly); planes = collect(halfspaces(hr))
    potential_faces = collect(combinations(1:n, 4)); thread_sets = [Set{NTuple{4, Int}}() for _ in 1:nthreads()]
    @threads for face_indices in potential_faces
        face_points = P[collect(face_indices), :]
        on_boundary = any(plane -> all(iszero, face_points * plane.a .- plane.β), planes)
        if !on_boundary; push!(thread_sets[threadid()], Tuple(sort(collect(face_indices)))); end
    end
    return union(thread_sets...)
end

function simplices_intersect_sat_cpu_4d(s1, s2)
    volume(intersect(polyhedron(vrep(s1)), polyhedron(vrep(s2)))) != 4
end

# --- Main Processing Functions ---

function process_polytope(initial_vertices_int::Matrix{Int}, id::Int, run_idx::Int, total_in_run::Int, config::Config)
    dim = size(initial_vertices_int, 2)
    buf = IOBuffer()
    timings = Vector{Pair{String, Float64}}()
    t_start_total = time_ns()
    initial_vertices = Rational{BigInt}.(initial_vertices_int)

    function log_verbose(msg...; is_display::Bool=false)
        timestamp = Dates.format(now(), "HH:MM:SS")
        s_msg = if is_display; sprint(show, "text/plain", msg[1]); else; join(msg, " "); end
        full_msg = "[$timestamp] " * s_msg
        println(buf, full_msg)
        if config.terminal_output == "verbose"; println(stdout, full_msg); flush(stdout); end
    end
    
    log_verbose("Processing $(dim)D Polytope #$id")
    if config.show_initial_vertices; log_verbose("Initial vertices provided:"); log_verbose(initial_vertices_int, is_display=true); end

    log_verbose("Step 1: Finding all lattice points..."); t_start = time_ns()
    P = dim == 3 ? findAllLatticePointsInHull_3d(initial_vertices) : findAllLatticePointsInHull_4d(initial_vertices)
    push!(timings, "Find all lattice points" => (time_ns() - t_start) / 1e9); num_lattice_points = size(P, 1)
    log_verbose("-> Found $num_lattice_points lattice points. Step 1 complete.\n")
    if config.terminal_output == "minimal"; update_line("($(@sprintf("%5d / %-5d", run_idx, total_in_run))): |P|=$num_lattice_points..."); end
    
    simplex_search_type = config.find_all_simplices ? "non-degenerate" : "unimodular"
    log_verbose("Step 2: Searching for $simplex_search_type $(dim)-simplices..."); t_start = time_ns()
    S_indices = dim == 3 ? all_simplices_in_3d(P, unimodular_only=!config.find_all_simplices) : all_simplices_in_4d(P, unimodular_only=!config.find_all_simplices)
    push!(timings, "Find all $simplex_search_type simplices" => (time_ns() - t_start) / 1e9); num_simplices_found = length(S_indices)
    log_verbose("-> Found $num_simplices_found simplices. Step 2 complete.\n")
    if config.terminal_output == "minimal"; update_line("($(@sprintf("%5d / %-5d", run_idx, total_in_run))): |P|=$num_lattice_points, |S|=$num_simplices_found..."); end

    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        msg = "no $simplex_search_type simplices exist"
        # --- CHANGE #1 START: Removed ID from this line ---
        minimal_log = @sprintf("(%5d / %-5d): |P|=%-5d |S|=%-7d -> \u001b[31m%s\u001b[0m", run_idx, total_in_run, num_lattice_points, num_simplices_found, msg)
        # --- CHANGE #1 END ---
        verbose_log = String(take!(buf)) * "\nNo simplices available. Cannot proceed."
        if config.terminal_output == "verbose"
            println(stdout, verbose_log)
        end
        return ProcessResult(id, 0, total_time, num_lattice_points, num_simplices_found, verbose_log, minimal_log, [])
    end

    log_verbose("Step 3: Precomputing open internal faces..."); t_start = time_ns()
    open_faces_set = dim == 3 ? precompute_open_faces_3d(P) : precompute_open_faces_4d(P)
    push!(timings, "Precompute open faces (parallel)" => (time_ns() - t_start) / 1e9)
    log_verbose("-> Found $(length(open_faces_set)) unique open faces. Step 3 complete.\n")

    num_simplices = length(S_indices); cnf = Vector{Vector{Int}}(); push!(cnf, collect(1:num_simplices))

    log_verbose("Step 4a: Generating intersection clauses..."); t_start = time_ns()
    
    intersection_clauses = let n_simplices = num_simplices
        if dim == 3 && config.intersection_backend == "gpu" && isdefined(Main, :GPUIntersection)
            log_verbose("     Using 3D GPU backend...")
            Main.GPUIntersection.get_intersecting_pairs_gpu(P, S_indices)
        elseif dim == 4 && config.intersection_backend == "gpu" && isdefined(Main, :GPUIntersection4D)
            log_verbose("     Using 4D GPU backend...")
            Main.GPUIntersection4D.get_intersecting_pairs_gpu_4d(P, S_indices)
        else
            if config.intersection_backend == "gpu"
                 log_verbose("     WARNING: GPU backend requested but not available. Check that CUDA is functional and 'gpu_intersection_$(dim)d.jl' is present. Using CPU.")
            else
                 log_verbose("     Using CPU backend.")
            end
            intersect_func = dim == 3 ? tetrahedra_intersect_sat_cpu_3d : simplices_intersect_sat_cpu_4d
            thread_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()]; next_i1 = Threads.Atomic{Int}(1)
            @threads for _ in 1:nthreads()
                tid = threadid()
                while true; i1 = Threads.atomic_add!(next_i1, 1); if i1 > n_simplices break end
                    if config.terminal_output == "verbose" && tid == 1 && (i1 % 50 == 0 || i1 == n_simplices)
                        update_line("[$(Dates.format(now(), "HH:MM:SS"))]      ... checking intersections (outer loop): $i1 / $n_simplices")
                    end
                    for i2 in (i1 + 1):n_simplices
                        if intersect_func(P[collect(S_indices[i1]), :], P[collect(S_indices[i2]), :]); push!(thread_clauses[tid], [-(i1), -(i2)]); end
                    end
                end
            end
            vcat(thread_clauses...)
        end
    end
    if config.terminal_output == "verbose"; update_line(""); println(stdout); end
    push!(timings, "Generate intersection clauses" => (time_ns() - t_start) / 1e9); append!(cnf, intersection_clauses); log_verbose("-> Found $(length(intersection_clauses)) intersection clauses. Step 4a complete.\n")

    log_verbose("Step 4b: Generating face-covering clauses..."); t_start = time_ns(); face_dim = dim
    face_clauses = let n_simplices = num_simplices
        thread_face_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()]; next_simplex_idx = Threads.Atomic{Int}(1)
        @threads for _ in 1:nthreads()
            tid = threadid()
            while true; i = Threads.atomic_add!(next_simplex_idx, 1); if i > n_simplices break end
                if config.terminal_output == "verbose" && tid == 1 && (i % 100 == 0 || i == n_simplices)
                    update_line("[$(Dates.format(now(), "HH:MM:SS"))]      ... checking for face covers: $i / $n_simplices")
                end
                for face_indices in combinations(S_indices[i], face_dim)
                    canonical_face = Tuple(sort(collect(face_indices)))
                    if canonical_face in open_faces_set
                        coverers = [j for (j, s2) in enumerate(S_indices) if i != j && issubset(canonical_face, s2)]
                        push!(thread_face_clauses[tid], vcat([-i], coverers))
                    end
                end
            end
        end
        vcat(thread_face_clauses...)
    end
    if config.terminal_output == "verbose"; update_line(""); println(stdout); end
    append!(cnf, face_clauses)
    push!(timings, "Generate face-covering clauses" => (time_ns() - t_start) / 1e9)
    log_verbose("-> Found $(length(face_clauses)) face-covering clauses. Step 4b complete.\n")
    
    log_verbose("Step 5: Handing SAT problem to solver ($(config.solver))..."); log_verbose("     Problem details: $(num_simplices) variables, $(length(cnf)) clauses.")
    log_verbose("     This step may take a long time if the problem is complex.")
    if config.terminal_output == "minimal"; update_line("($(@sprintf("%5d / %-5d", run_idx, total_in_run))): |P|=$num_lattice_points, |S|=$num_simplices_found, solving..."); end
    
    t_start_solve = time_ns(); solutions = []; num_solutions = 0
    solver_used = config.solver
    if solver_used == "CryptoMiniSat" && !CMS_LOADED[]; log_verbose("Warning: CryptoMiniSat not available, falling back to PicoSAT."); solver_used = "PicoSAT"; end

    solver_func = solver_used == "CryptoMiniSat" ? CryptoMiniSat : PicoSAT
    if config.solution_reporting == "first"
        solution = solver_func.solve(cnf); if solution isa Vector{Int}; num_solutions = 1; push!(solutions, solution); end
    else
        for solution in solver_func.itersolve(cnf); num_solutions += 1; if config.solution_reporting == "all"; push!(solutions, solution); end; end
    end

    log_verbose("-> SAT solver finished. Step 5 complete.")
    push!(timings, "Solve SAT problem" => (time_ns() - t_start_solve) / 1e9); total_time = (time_ns() - t_start_total) / 1e9; push!(timings, "Total execution time" => total_time)
    log_verbose("\n$(num_solutions) valid triangulation(s) found.")
    
    first_solution_simplices = Vector{Matrix{Int}}()
    if num_solutions > 0 && !isempty(solutions)
        first_sol_indices = findall(l -> l > 0, first(solutions))
        first_solution_simplices = [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in first_sol_indices]
    end

    if config.show_solution_simplices && !isempty(first_solution_simplices)
        log_verbose("\nDisplaying first valid triangulation:"); for s in first_solution_simplices; log_verbose(s, is_display=true); end
    end
    
    if num_solutions > 0 && !isempty(config.plot_range) && !isempty(config.plotter_script) && (id in parse_range(config.plot_range, 1_000_000))
        log_verbose("\nStep 6: Plotting results...")
        if dim == 3
            temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(first_solution_simplices)); close(temp_io); run(`python $(config.plotter_script) $(temp_path)`); finally rm(temp_path, force=true); end
        else
            initial_poly = polyhedron(vrep(initial_vertices)); boundary_planes = collect(halfspaces(hrep(initial_poly)))
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
    result_str = num_solutions > 0 ? @sprintf("\u001b[32mfound %d solution(s)\u001b[0m in %.2f s", num_solutions, total_time) : @sprintf("\u001b[31mno solution exists\u001b[0m, searched for %.2f s", total_time)
    # --- CHANGE #2 START: Removed ID from this line ---
    minimal_log = @sprintf("(%5d / %-5d): |P|=%-5d |S|=%-7d -> %s", run_idx, total_in_run, num_lattice_points, num_simplices_found, result_str)
    # --- CHANGE #2 END ---
    
    return ProcessResult(id, num_solutions, total_time, num_lattice_points, num_simplices_found, String(take!(buf)), minimal_log, first_solution_simplices)
end

function run_processing(polytopes::Vector{Matrix{Int}}, dim::Int, config::Config, range_to_process, log_stream)
    indices_to_process = collect(range_to_process)

    # --- New Sorting Logic ---
    if config.sort_by in ["P", "S"]
        println("Pre-calculating metrics for sorting by '$(config.sort_by)'...")
        
        metrics = []
        total = length(indices_to_process)
        for (i, p_idx) in enumerate(indices_to_process)
            update_line("Sorting pre-calculation: $i / $total")
            initial_vertices = Rational{BigInt}.(polytopes[p_idx])
            
            # Calculate P (number of lattice points)
            P = dim == 3 ? findAllLatticePointsInHull_3d(initial_vertices) : findAllLatticePointsInHull_4d(initial_vertices)
            num_points = size(P, 1)

            if config.sort_by == "P"
                push!(metrics, (p_idx, num_points))
            elseif config.sort_by == "S"
                # Calculate S (number of simplices)
                S_indices = dim == 3 ? all_simplices_in_3d(P, unimodular_only=!config.find_all_simplices) : all_simplices_in_4d(P, unimodular_only=!config.find_all_simplices)
                num_simplices = length(S_indices)
                push!(metrics, (p_idx, num_simplices))
            end
        end
        println() # New line after update_line
        
        # Sort indices based on the calculated metric (the second element of the tuple)
        sort!(metrics, by = x -> x[2])
        indices_to_process = [m[1] for m in metrics]
    end
    # --- End of New Sorting Logic ---

    if config.processing_order == "reversed"; reverse!(indices_to_process); elseif config.processing_order == "random"; shuffle!(indices_to_process); end

    if config.terminal_output != "none"
        println("Run started at:                          $(Dates.format(now(), "HH:MM:SS"))")
        println("Number of threads:                       $(nthreads())")
        println("Detected Dimension:                      $(dim)")
        println("Solver requested:                        $(config.solver)")
        println("Solve mode:                              $(config.solution_reporting)")
        println("Input file:                              $(config.polytopes_file)")
        println("Number of polytopes found:               $(length(indices_to_process))")
        println("Processing range:                        $(range_to_process)")
        println("Restricting to unimodular simplices:     $(!config.find_all_simplices)")
        if config.file_output != "none" && config.log_file != ""
            println("Writing to log file:                     $(config.log_file)")
        end
        println("Intersection backend selected:           $(config.intersection_backend)")
        println("")
    end
    
    if config.terminal_output == "minimal" && config.terminal_mode == "single-line"
        println("\n\n\n") # Reserve 4 lines: 3 for summary, 1 for progress
    end
    
    t_start_run = time(); total_solutions_found = 0; triangulations_found_count = 0; non_triangulatable_count = 0

    for (i, p_idx) in enumerate(indices_to_process)
        result = process_polytope(polytopes[p_idx], p_idx, i, length(indices_to_process), config)
        if result.num_solutions_found > 0; triangulations_found_count += 1; else non_triangulatable_count += 1; end
        total_solutions_found += result.num_solutions_found
        
        if config.terminal_output == "minimal"
            if config.terminal_mode == "single-line"
                elapsed_time = time() - t_start_run

                # Move cursor to the top of the summary block (from the 4th line to the 1st)
                print(stdout, "\u001b[3A") # Move cursor up 3 lines

                # Print summary lines, clearing each line to prevent artifacts
                @printf(stdout, "\r%-40s %.2fs\u001b[K\n", "Elapsed Time:", elapsed_time)
                # --- CHANGE START: Add color codes around the numbers ---
                @printf(stdout, "\r%-40s \u001b[32m%d\u001b[0m\u001b[K\n", "Polytopes with Solutions:", triangulations_found_count)
                @printf(stdout, "\r%-40s \u001b[31m%d\u001b[0m\u001b[K\n", "Non-Triangulatable:", non_triangulatable_count)
                # --- CHANGE END ---

                # Print the final result for the completed polytope on the 4th line
                print(stdout, "\r" * result.minimal_log * "\u001b[K")
                flush(stdout)
            else # This handles the "multi-line" case
                println(stdout, result.minimal_log)
            end
        end

        if !isnothing(log_stream)
            log_content = config.file_output == "verbose" ? result.verbose_log : result.minimal_log * "\n"
            print(log_stream, log_content); flush(log_stream)
        end
    end
    
    if config.terminal_output == "minimal" && config.terminal_mode == "single-line"; println(); end
    total_time_run = time() - t_start_run
    
    avg_solutions_str = ""
    if config.solution_reporting in ["all", "count_only"] && !isempty(indices_to_process)
        avg_solutions_str = @sprintf("Average Solutions/Polytope:      %.2f\n", total_solutions_found / length(indices_to_process))
    end

    summary_str = """

    Run finished: $(Dates.format(now(), "HH:MM:SS"))

    ----------------------------------------
    Run Summary
    ----------------------------------------
    Total Polytopes Processed:       $(length(indices_to_process))
    Polytopes with Solutions:        \u001b[32m$triangulations_found_count\u001b[0m
    Non-Triangulatable:              \u001b[31m$non_triangulatable_count\u001b[0m
    $(avg_solutions_str)Total Run Time:                  $(@sprintf("%.2f", total_time_run)) seconds
    ----------------------------------------
    """
    print(stdout, summary_str)
    if !isnothing(log_stream); print(log_stream, replace(summary_str, r"\u001b\[\d+m" => "")); end
end

function main()
    config_path = isempty(ARGS) ? "config.toml" : ARGS[1]
    if !isfile(config_path); println(stderr, "Error: Config file not found at '$config_path'"); return; end
    config = load_config(config_path)

    # --- Pre-run Checks for Available Packages ---
    if config.intersection_backend == "gpu"
        if !CUDA_PACKAGES_LOADED[]; @warn "GPU backend requested, but CUDA not loaded. Falling back to 'cpu'."; config.intersection_backend = "cpu";
        elseif !CUDA.functional(); @warn "CUDA loaded, but no functional GPU found. Falling back to 'cpu'."; config.intersection_backend = "cpu";
        end
    end
    if config.solver == "CryptoMiniSat" && !CMS_LOADED[]; @warn "CryptoMiniSat solver requested, but not loaded. Falling back to PicoSAT."; config.solver = "PicoSAT"; end

    log_stream = nothing
    if !isempty(config.log_file); try log_stream = open(config.log_file, "a"); println(log_stream, "\n\n" * "#"^80, "\n# New Run Started at $(now())\n" * "#"^80); catch e; println(stderr, "Error opening log file: $e"); return; end; end

    try
        polytopes = read_polytopes_from_file(config.polytopes_file)
        if isempty(polytopes); println(stderr, "Error: No polytopes loaded from '$(config.polytopes_file)'."); return; end
        range_to_process = parse_range(config.process_range, length(polytopes))
        if isempty(range_to_process) || !checkbounds(Bool, polytopes, range_to_process); println(stderr, "Error: 'process_range' is out of bounds."); return; end

        # --- Dimension Detection ---
        dim = size(polytopes[first(range_to_process)], 2)
        if !(dim in [3, 4]); println(stderr, "Error: Detected dimension is $dim. Only 3D and 4D are supported."); return; end
        
        # This is the main dispatcher for the entire run.
        run_processing(polytopes, dim, config, range_to_process, log_stream)
        
    finally
        !isnothing(log_stream) && close(log_stream)
    end
end

main()

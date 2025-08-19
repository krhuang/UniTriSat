using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf
using Base.Threads
using TOML
using Random
using OpenCL
using CUDA
using StaticArrays
using CUDA.Adapt

include("gpu_backends/cuda_backend.jl")
include("gpu_backends/opencl_backend.jl")

# --- Configuration and Data Structures ---

struct Config
    polytopes_file::String
    log_file::String
    process_range::String
    processing_order::String
    intersection_backend::String
    gpu_type::String
    remove_origin_simplices::Bool
    terminal_output::String
    file_output::String
    show_initial_vertices::Bool
    show_solution_simplices::Bool
    show_timing_summary::Bool
    solver::String
    geometry_tolerance::Float64
end

struct ProcessResult
    id::Int
    is_valid_triangulation::Bool
    total_time::Float64
    num_lattice_points::Int
    num_simplices_found::Int
    verbose_log::String
    minimal_log::String
end

function load_config(filepath::String)
    config_dict = TOML.parsefile(filepath)
    files = get(config_dict, "files", Dict())
    run_settings = get(config_dict, "run_settings", Dict())
    output_levels = get(config_dict, "output_levels", Dict())
    verbose_options = get(config_dict, "verbose_options", Dict())
    solver_options = get(config_dict, "solver_options", Dict())
    params = get(config_dict, "parameters", Dict())

    backend = get(run_settings, "intersection_backend", "cpu")
    if backend ∉ ["cpu", "gpu"]
        @warn "Invalid 'intersection_backend' value: '$(backend)'. Defaulting to 'cpu'."
        backend = "cpu"
    end

    gpu_type = get(run_settings, "gpu_type", "cuda")
    if backend == "gpu"
        if gpu_type ∉ ["cuda", "opencl", "oneapi", "amd"]
            @warn "Invalid 'gpu_type' value: '$(gpu_type)'. Defaulting to 'cuda'."
            gpu_type = "cuda"
        end

        if gpu_type == "cuda" && !CUDA.functional()
            @warn "CUDA is not available on this system. Falling back to 'cpu' backend."
            backend = "cpu"
        end
    end

    return Config(
        get(files, "polytopes_file", "polytopes.txt"),
        get(files, "log_file", ""),
        get(run_settings, "process_range", "1-"),
        get(run_settings, "processing_order", "normal"),
        backend,
        gpu_type,   # NEW
        get(run_settings, "remove_origin_simplices", false),
        get(output_levels, "terminal_output", "verbose"),
        get(output_levels, "file_output", "verbose"),
        get(verbose_options, "show_initial_vertices", true),
        get(verbose_options, "show_solution_simplices", true),
        get(verbose_options, "show_timing_summary", true),
        get(solver_options, "solver", "PicoSAT"),
        get(params, "geometry_tolerance", 1e-8)
    )
end

# --- Helper Functions ---

function findAllLatticePointsInHull(vertices::Matrix{Int}, config::Config)
    poly = polyhedron(vrep(vertices));
    hr = hrep(poly)
    min_coords = floor.(Int, minimum(vertices, dims=1));
    max_coords = ceil.(Int, maximum(vertices, dims=1))
    lattice_points = Vector{Vector{Int}}()
    for iz in min_coords[3]:max_coords[3], iy in min_coords[2]:max_coords[2], ix in min_coords[1]:max_coords[1]
        point = [ix, iy, iz]
        if all(hr.A * point .<= hr.b .+ config.geometry_tolerance)
            push!(lattice_points, point)
        end
    end
    return isempty(lattice_points) ?
        Matrix{Int}(undef, 0, 3) : vcat(lattice_points'...)
end

function all_unimodular_simplices_in(P::Matrix{Int})
    n = size(P, 1);
    simplex_indices = Vector{NTuple{4, Int}}()
    if n < 4 return simplex_indices end
    for inds in combinations(1:n, 4)
        p0, p1, p2, p3 = P[inds[1], :], P[inds[2], :], P[inds[3], :], P[inds[4], :]
        M = vcat((p1 - p0)', (p2 - p0)', (p3 - p0)');
        d = round(Int, det(M))
        if abs(d) == 1; push!(simplex_indices, Tuple(inds));
        end
    end
    return simplex_indices
end

function precompute_open_faces(P::Matrix{Int}, config::Config)
    n = size(P, 1)
    if n < 3 return Set{NTuple{3, Int}}() end
    poly = polyhedron(vrep(P));
    hr = hrep(poly)
    planes = collect(halfspaces(hr)); potential_faces = collect(combinations(1:n, 3))
    thread_sets = [Set{NTuple{3, Int}}() for _ in 1:nthreads()]
    @threads for face_indices in potential_faces
        face_points = P[face_indices, :]
        on_boundary = false
        for plane in planes
            if all(abs.(face_points * plane.a .- plane.β) .<= config.geometry_tolerance)
                on_boundary = true; break
            end
        end
        if !on_boundary;
            push!(thread_sets[threadid()], Tuple(sort(collect(face_indices))));
        end
    end
    return union(thread_sets...)
end

# --- Intersection Logic: CPU ---

function tetrahedra_intersect_volume(tetra1_verts::Matrix, tetra2_verts::Matrix)
    t1 = Float64.(tetra1_verts);
    t2 = Float64.(tetra2_verts)
    vol1 = abs(dot(t1[1,:] - t1[4,:], cross(t1[2,:] - t1[4,:], t1[3,:] - t1[4,:]))) / 6.0
    vol2 = abs(dot(t2[1,:] - t2[4,:], cross(t2[2,:] - t2[4,:], t2[3,:] - t2[4,:]))) / 6.0
    if vol1 < 1e-9 || vol2 < 1e-9; return false; end
    axes = []; append!(axes, _get_outward_face_normals(t1));
    append!(axes, _get_outward_face_normals(t2))
    edges1 = [t1[j,:] - t1[i,:] for (i, j) in combinations(1:4, 2)];
    edges2 = [t2[j,:] - t2[i,:] for (i, j) in combinations(1:4, 2)]
    for e1 in edges1, e2 in edges2
        axis = cross(e1, e2)
        if norm(axis) > 1e-9;
            push!(axes, axis);
        end
    end
    for axis in axes
        min1, max1 = _project(t1, axis);
        min2, max2 = _project(t2, axis)
        if max1 <= min2 || max2 <= min1;
            return false;
        end
    end
    return true
end

function _project(vertices::Matrix, axis::Vector); projections = vertices*axis; return minimum(projections), maximum(projections);
end

function _get_outward_face_normals(vertices::Matrix)
    normals = []; face_indices_list = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    for face_indices in face_indices_list
        p0,p1,p2 = vertices[face_indices[1],:], vertices[face_indices[2],:], vertices[face_indices[3],:]
        normal = cross(p1 - p0, p2 - p0)
        p_fourth = vertices[first(setdiff(1:4, face_indices)), :];
        if dot(normal, p_fourth - p0) > 0; normal = -normal;
        end
        if norm(normal) > 1e-9; push!(normals, normal);
        end
    end
    return normals
end

# --- Utility Functions ---

function read_polytopes_from_file(filepath::String)
    polytopes = Vector{Matrix{Int}}()
    current_vertices = Vector{Vector{Int}}()
    for line in eachline(filepath)
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            if !isempty(current_vertices); push!(polytopes, vcat(current_vertices'...)); current_vertices = [];
            end
            continue
        end
        push!(current_vertices, parse.(Int, split(line)))
    end
    if !isempty(current_vertices);
        push!(polytopes, vcat(current_vertices'...));
    end
    return polytopes
end

function parse_range(s::String, max_index::Int)
    s = strip(s)
    if s == "-" || isempty(s); return 1:max_index; end
    if endswith(s, "-"); return parse(Int, s[1:end-1]):max_index; end
    if contains(s, "-");
        parts = parse.(Int, split(s, "-")); return parts[1]:parts[2];
    end
    idx = parse(Int, s);
    return idx:idx
end

function update_line(message::String)
    print(stdout, "\r" * message * "\u001b[K")
    flush(stdout)
end

# --- Main Processing Function ---

function process_polytope(initial_vertices::Matrix{Int}, id::Int, config::Config)
    buf = IOBuffer()
    timings = Vector{Pair{String, Float64}}()
    t_start_total = time_ns()

    function log_verbose(msg...; is_display::Bool=false)
        if is_display
            show(buf, "text/plain", msg[1]);
            println(buf)
        else
            println(buf, msg...)
        end
    end

    log_verbose("\n" * "="^80, "\nProcessing Polytope #$id\n" * "="^80)
    if config.show_initial_vertices
        log_verbose("Initial vertices provided:");
        log_verbose(initial_vertices, is_display=true)
    end

    t_start = time_ns();
    P = findAllLatticePointsInHull(initial_vertices, config)
    push!(timings, "Find all lattice points" => (time_ns() - t_start) / 1e9)
    num_lattice_points = size(P, 1)
    log_verbose("Found $num_lattice_points lattice points in the hull.\n")
    if config.terminal_output == "minimal" update_line("($id): |P|=$num_lattice_points...") end

    t_start = time_ns();
    S_indices = all_unimodular_simplices_in(P)
    push!(timings, "Find all unimodular simplices" => (time_ns() - t_start) / 1e9)
    num_simplices_found = length(S_indices)
    log_verbose("Number of unimodular simplices found: $num_simplices_found")
    if config.terminal_output == "minimal" update_line("($id): |P|=$num_lattice_points, |S|=$num_simplices_found...") end

    if config.remove_origin_simplices
        log_verbose("\nConfiguration: Removing simplices containing the origin.")
        origin_idx = findfirst(row -> all(row .== 0), eachrow(P))
        if !isnothing(origin_idx)
            initial_count = length(S_indices)
            filter!(s_tuple -> origin_idx ∉ s_tuple, S_indices)
            log_verbose("Removed $(initial_count - length(S_indices)) simplices.")
        end
    end
    log_verbose("Using $(length(S_indices)) simplices for SAT problem.\n")

    if isempty(S_indices)
        total_time = (time_ns() - t_start_total) / 1e9
        minimal_log = "($id): |P|=$num_lattice_points, |S|=0, no triangulation exists, searched for $(round(total_time, digits=2)) seconds"
        verbose_log_str = String(take!(buf)) * "\nNo unimodular simplices available. Cannot proceed."
        if config.terminal_output == "verbose"
            print(stdout, verbose_log_str)
        end
        return ProcessResult(id, false, total_time, num_lattice_points, num_simplices_found, verbose_log_str, minimal_log)
    end

    log_verbose("[$(Dates.format(now(), "HH:MM:SS"))] Precomputing open faces...")
    t_start = time_ns();
    open_faces_set = precompute_open_faces(P, config)
    push!(timings, "Precompute open faces (parallel)" => (time_ns() - t_start) / 1e9)
    log_verbose("Found $(length(open_faces_set)) unique open faces.\n")

    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}();
    push!(cnf, collect(1:num_simplices))

    log_verbose("[$(Dates.format(now(), "HH:MM:SS"))] Generating intersection clauses using $(uppercase(config.intersection_backend)) backend...")
    t_start = time_ns()

    intersection_clauses = Vector{Vector{Int}}()
    if config.intersection_backend == "gpu"
        if config.gpu_type == "cuda" 
            intersection_clauses = CUDABackend.get_intersecting_pairs_cuda(P, S_indices)
            push!(timings, "Generate intersection clauses (CUDA)" => (time_ns() - t_start) / 1e9)

        elseif config.gpu_type == "opencl"
            intersection_clauses = OpenCLBackend.get_intersecting_pairs_opencl(P, S_indices)
            push!(timings, "Generate intersection clauses (CUDA)" => (time_ns() - t_start) / 1e9)

        elseif config.gpu_type == "oneapi"
            error("oneAPI backend not implemented yet. Please implement get_intersecting_pairs_oneapi().")

        elseif config.gpu_type == "amd"
            error("AMDGPU backend not implemented yet. Please implement get_intersecting_pairs_amd().")
        end
        
    else # "cpu"
        thread_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()];
        next_i1 = Threads.Atomic{Int}(1)
        @threads for _ in 1:nthreads()
            tid = threadid();
            while true
                i1 = Threads.atomic_add!(next_i1, 1);
                if i1 > num_simplices break end
                for i2 in (i1 + 1):num_simplices
                    s1_verts = P[collect(S_indices[i1]), :];
                    s2_verts = P[collect(S_indices[i2]), :]
                    if tetrahedra_intersect_volume(s1_verts, s2_verts);
                        push!(thread_clauses[tid], [-(i1), -(i2)]);
                    end
                end
            end
        end
        intersection_clauses = vcat(thread_clauses...);
        push!(timings, "Generate intersection clauses (CPU, parallel)" => (time_ns() - t_start) / 1e9)
    end
    append!(cnf, intersection_clauses)
    log_verbose("Number of intersection clauses: $(length(intersection_clauses))")

    log_verbose("[$(Dates.format(now(), "HH:MM:SS"))] Generating face-covering clauses...")
    t_start = time_ns()
    thread_face_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()];
    next_simplex_idx = Threads.Atomic{Int}(1)
    @threads for _ in 1:nthreads()
        tid = threadid();
        while true
            i = Threads.atomic_add!(next_simplex_idx, 1);
            if i > num_simplices break end
            s_verts_indices = S_indices[i]
            for face_indices in combinations(s_verts_indices, 3)
                canonical_face = Tuple(sort(collect(face_indices)))
                if canonical_face in open_faces_set
                    coverers = [j for (j, s2_verts_indices) in enumerate(S_indices) if i != j && issubset(canonical_face, s2_verts_indices)]
                    push!(thread_face_clauses[tid], vcat([-i], coverers))
                end
            end
        end
    end
    face_clauses = vcat(thread_face_clauses...);
    append!(cnf, face_clauses)
    push!(timings, "Generate face-covering clauses (parallel)" => (time_ns() - t_start) / 1e9)
    log_verbose("Number of face-covering clauses: $(length(face_clauses))")
    log_verbose("Total number of clauses: $(length(cnf))\n")

    log_verbose("[$(Dates.format(now(), "HH:MM:SS"))] Start solving with $(config.solver)...")
    if config.terminal_output == "minimal" update_line("($id): |P|=$num_lattice_points, |S|=$num_simplices_found, solving...") end
    t_start = time_ns();
    solution = PicoSAT.solve(cnf)
    push!(timings, "Solve SAT problem ($(config.solver))" => (time_ns() - t_start) / 1e9)
    total_time = (time_ns() - t_start_total) / 1e9
    push!(timings, "Total execution time" => total_time)

    poly_volume = volume(polyhedron(vrep(initial_vertices)));
    expected_simplices = round(Int, 6 * poly_volume)
    is_satisfiable = solution isa Vector{Int}
    is_valid_triangulation = false
    if is_satisfiable
        chosen_indices = findall(l -> l > 0, solution)
        is_valid_triangulation = length(chosen_indices) == expected_simplices
    end

    if is_satisfiable
        println(buf, "\nA satisfying assignment was found!")
        println(buf, "\nNumber of simplices in solution: $(length(chosen_indices))")
        println(buf, "Expected number for a full triangulation: $expected_simplices")
        if is_valid_triangulation;
            println(buf, "Assertion successful: The solution size matches the expected volume.")
        else;
            println(buf, "ERROR: The solution found does NOT form a valid triangulation.");
        end
        if config.show_solution_simplices && is_valid_triangulation
            println(buf, "\nDisplaying simplices for the valid triangulation:")
            for simplex_var_index in chosen_indices
                simplex_point_indices = S_indices[simplex_var_index]
                show(buf, "text/plain", P[collect(simplex_point_indices), :]);
                println(buf)
            end
        end
    else
        println(buf, "\nUNSATISFIABLE: No solution was found. Status: $solution")
    end

    if config.show_timing_summary
        println(buf, "\n--- Timing & Memory Summary for Polytope #$id ---")
        peak_ram_bytes = Sys.maxrss()
        for (operation, duration) in timings;
            println(buf, @sprintf("%-45s: %.4f seconds", operation, duration));
        end
        println(buf, @sprintf("%-45s: %.2f MiB", "Peak memory usage (Max RSS)", peak_ram_bytes / 1024^2))
    end

    minimal_log = "($id): |P|=$num_lattice_points, |S|=$num_simplices_found, " * (
        is_valid_triangulation ? "found a triangulation in $(round(total_time, digits=2)) seconds" :
        "no triangulation exists, searched for $(round(total_time, digits=2)) seconds"
    )

    return ProcessResult(id, is_valid_triangulation, total_time, num_lattice_points, num_simplices_found, String(take!(buf)), minimal_log)
end

# --- Main Execution Block ---

function main()
    config_path = isempty(ARGS) ? "config.toml" : ARGS[1]
    if !isfile(config_path)
        println(stderr, "Error: Config file not found at '$config_path'");
        return
    end
    config = load_config(config_path)

    log_stream = nothing
    if !isempty(config.log_file)
        try
            log_stream = open(config.log_file, "a")
            println(log_stream, "\n\n" * "#"^80, "\n# New Run Started at $(Dates.format(now(), "YYYY-mm-dd HH:MM:SS"))\n" * "#"^80)
        catch e
            println(stderr, "Error opening log file: $e");
            return
        end
    end

    try
        polytopes = read_polytopes_from_file(config.polytopes_file)
        range_to_process = parse_range(config.process_range, length(polytopes))

        if range_to_process.start < 1 || range_to_process.stop > length(polytopes)
            println(stderr, "Error: 'process_range' in config is out of bounds.");
            return
        end
        indices_to_process = collect(range_to_process)
        if config.processing_order == "reversed";
            reverse!(indices_to_process);
        elseif config.processing_order == "random"; shuffle!(indices_to_process);
        end

        println("Julia running with $(nthreads()) threads.")
        println("Processing order: $(config.processing_order)")
        if config.intersection_backend == "gpu"
             println("Using GPU for intersection calculations.")
        end

        for i in indices_to_process
            result = process_polytope(polytopes[i], i, config)

            if config.terminal_output == "verbose";
                print(stdout, result.verbose_log);
            elseif config.terminal_output == "minimal"; update_line(result.minimal_log); println(stdout);
            end

            if !isnothing(log_stream)
                if config.file_output == "verbose";
                    print(log_stream, result.verbose_log);
                elseif config.file_output == "minimal"; println(log_stream, result.minimal_log);
                end
                flush(log_stream)
            end
        end
    finally
        if !isnothing(log_stream);
            close(log_stream);
        end
    end
end

main()

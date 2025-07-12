using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf
using Base.Threads

# --- Helper Functions (Geometry, Precomputation) ---
# These are unchanged from the previous version.

"""
    findAllLatticePointsInHull(vertices::Matrix{Int})
"""
function findAllLatticePointsInHull(vertices::Matrix{Int})
    println("Finding all lattice points inside the convex hull of the input vertices...")
    poly = polyhedron(vrep(vertices))
    hr = hrep(poly)
    min_coords = floor.(Int, minimum(vertices, dims=1))
    max_coords = ceil.(Int, maximum(vertices, dims=1))
    lattice_points = Vector{Vector{Int}}()
    tol = 1e-8
    for iz in min_coords[3]:max_coords[3], iy in min_coords[2]:max_coords[2], ix in min_coords[1]:max_coords[1]
        point = [ix, iy, iz]
        if all(hr.A * point .<= hr.b .+ tol)
            push!(lattice_points, point)
        end
    end
    return vcat(lattice_points'...)
end

"""
    all_unimodular_simplices_in(P)
"""
function all_unimodular_simplices_in(P::Matrix{Int})
    n = size(P, 1)
    simplex_indices = Vector{NTuple{4, Int}}()
    for inds in combinations(1:n, 4)
        p0, p1, p2, p3 = P[inds[1], :], P[inds[2], :], P[inds[3], :], P[inds[4], :]
        M = vcat((p1 - p0)', (p2 - p0)', (p3 - p0)')
        d = round(Int, det(M))
        if abs(d) == 1
            push!(simplex_indices, Tuple(inds))
        end
    end
    return simplex_indices
end

"""
    precompute_open_faces(P; tol=1e-8)
"""
function precompute_open_faces(P::Matrix{Int}; tol=1e-8)
    n = size(P, 1)
    poly = polyhedron(vrep(P))
    hr = hrep(poly)
    planes = collect(halfspaces(hr))
    potential_faces = collect(combinations(1:n, 3))
    thread_sets = [Set{NTuple{3, Int}}() for _ in 1:nthreads()]
    @threads for face_indices in potential_faces
        tid = threadid()
        face_points = P[face_indices, :]
        on_boundary = false
        for plane in planes
            if all(abs.(face_points * plane.a .- plane.β) .<= tol)
                on_boundary = true
                break
            end
        end
        if !on_boundary
            push!(thread_sets[tid], Tuple(sort(collect(face_indices))))
        end
    end
    return union(thread_sets...)
end

function tetrahedra_intersect_volume(tetra1_verts::Matrix, tetra2_verts::Matrix)
    t1 = Float64.(tetra1_verts); t2 = Float64.(tetra2_verts)
    vol1 = abs(dot(t1[1,:] - t1[4,:], cross(t1[2,:] - t1[4,:], t1[3,:] - t1[4,:]))) / 6.0
    vol2 = abs(dot(t2[1,:] - t2[4,:], cross(t2[2,:] - t2[4,:], t2[3,:] - t2[4,:]))) / 6.0
    if vol1 < 1e-9 || vol2 < 1e-9; return false; end
    axes = []; append!(axes, _get_outward_face_normals(t1)); append!(axes, _get_outward_face_normals(t2))
    edges1 = [t1[j,:] - t1[i,:] for (i, j) in combinations(1:4, 2)]; edges2 = [t2[j,:] - t2[i,:] for (i, j) in combinations(1:4, 2)]
    for e1 in edges1, e2 in edges2
        axis = cross(e1, e2)
        if norm(axis) > 1e-9; push!(axes, axis); end
    end
    for axis in axes
        min1, max1 = _project(t1, axis); min2, max2 = _project(t2, axis)
        if max1 <= min2 || max2 <= min1; return false; end
    end
    return true
end

function _project(vertices::Matrix, axis::Vector)
    projections = vertices * axis
    return minimum(projections), maximum(projections)
end

function _get_outward_face_normals(vertices::Matrix)
    normals = []; face_indices_list = collect(combinations(1:4, 3))
    for face_indices in face_indices_list
        p0, p1, p2 = vertices[face_indices[1], :], vertices[face_indices[2], :], vertices[face_indices[3], :]
        normal = cross(p1 - p0, p2 - p0)
        p_fourth = vertices[first(setdiff(1:4, face_indices)), :];
        if dot(normal, p_fourth - p0) > 0; normal = -normal; end
        if norm(normal) > 1e-9; push!(normals, normal); end
    end
    return normals
end


# --- NEW File and CLA Parsing Functions ---

"""
    read_polytopes_from_file(filepath::String)

Parses a text file where polytopes are defined by their vertices,
with each polytope separated by a blank line.
"""
function read_polytopes_from_file(filepath::String)
    polytopes = Vector{Matrix{Int}}()
    current_vertices = Vector{Vector{Int}}()
    
    for line in eachline(filepath)
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            if !isempty(current_vertices)
                push!(polytopes, vcat(current_vertices'...))
                current_vertices = []
            end
            continue
        end
        
        parts = parse.(Int, split(line))
        push!(current_vertices, parts)
    end
    
    if !isempty(current_vertices)
        push!(polytopes, vcat(current_vertices'...))
    end
    
    return polytopes
end

"""
    parse_range(s::String) -> UnitRange{Int}

Parses a string like "3" or "1-5" into a range.
"""
function parse_range(s::String)
    if contains(s, "-")
        parts = parse.(Int, split(s, "-"))
        return parts[1]:parts[2]
    else
        idx = parse(Int, s)
        return idx:idx
    end
end


"""
    process_polytope(initial_vertices::Matrix{Int}, id::Int)

The main analysis pipeline for a single polytope.
"""
function process_polytope(initial_vertices::Matrix{Int}, id::Int)
    println("\n" * "="^80)
    println("Processing Polytope #$id")
    println("="^80)

    timings = Vector{Pair{String, Float64}}()
    t_start_total = time_ns()
    
    println("Initial vertices provided:"); display(initial_vertices)

    t_start = time_ns()
    P = findAllLatticePointsInHull(initial_vertices)
    push!(timings, "Find all lattice points" => (time_ns() - t_start) / 1e9)
    println("Found $(size(P, 1)) lattice points in the hull.\n")
    
    t_start = time_ns()
    S_indices = all_unimodular_simplices_in(P)
    push!(timings, "Find all unimodular simplices" => (time_ns() - t_start) / 1e9)
    println("Number of unimodular simplices found: $(length(S_indices))\n")
    
    if isempty(S_indices)
        println("No unimodular simplices found. Cannot proceed.")
        return
    end

    println("[$(Dates.format(now(), "HH:MM:SS"))] Precomputing open faces on $(nthreads()) threads...")
    t_start = time_ns()
    open_faces_set = precompute_open_faces(P)
    push!(timings, "Precompute open faces (parallel)" => (time_ns() - t_start) / 1e9)
    println("Found $(length(open_faces_set)) unique open faces.\n")
    
    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))

    println("[$(Dates.format(now(), "HH:MM:SS"))] Generating intersection clauses on $(nthreads()) threads...")
    t_start = time_ns()
    thread_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()]
    next_i1 = Threads.Atomic{Int}(1)
    
    @threads for _ in 1:nthreads()
        tid = threadid()
        while true
            i1 = Threads.atomic_add!(next_i1, 1)
            if i1 > num_simplices break end
            for i2 in (i1 + 1):num_simplices
                s1_verts = P[collect(S_indices[i1]), :]
                s2_verts = P[collect(S_indices[i2]), :]
                if tetrahedra_intersect_volume(s1_verts, s2_verts)
                    push!(thread_clauses[tid], [-(i1), -(i2)])
                end
            end
        end
    end
    
    intersection_clauses = vcat(thread_clauses...)
    append!(cnf, intersection_clauses)
    push!(timings, "Generate intersection clauses (parallel)" => (time_ns() - t_start) / 1e9)
    println("Number of intersection clauses: $(length(intersection_clauses))")
    
    println("[$(Dates.format(now(), "HH:MM:SS"))] Generating face-covering clauses on $(nthreads()) threads...")
    t_start = time_ns()
    thread_face_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()]
    next_simplex_idx = Threads.Atomic{Int}(1)

    @threads for _ in 1:nthreads()
        tid = threadid()
        while true
            i = Threads.atomic_add!(next_simplex_idx, 1)
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

    face_clauses = vcat(thread_face_clauses...)
    append!(cnf, face_clauses)
    push!(timings, "Generate face-covering clauses (parallel)" => (time_ns() - t_start) / 1e9)
    println("Number of face-covering clauses: $(length(face_clauses))")
    println("Total number of clauses: $(length(cnf))\n")

    println("[$(Dates.format(now(), "HH:MM:SS"))] Start solving with PicoSAT...\n")
    t_start = time_ns()
    solution = PicoSAT.solve(cnf)
    push!(timings, "Solve SAT problem (PicoSAT)" => (time_ns() - t_start) / 1e9)

    # Calculate expected volume from the convex hull of the *initial* vertices
    poly_volume = volume(polyhedron(vrep(initial_vertices)))
    expected_simplices = round(Int, 6 * poly_volume)

    if solution isa Vector{Int}
        println("A satisfying assignment was found!")
        chosen_indices = findall(l -> l > 0, solution)
        println("\nNumber of simplices in solution: $(length(chosen_indices))")
        println("Expected number for a full triangulation: $expected_simplices")
        
        @assert length(chosen_indices) == expected_simplices "The solution found does not form a valid triangulation."
        println("Assertion successful: The solution size matches the expected volume for a triangulation.")
        
        println("\nDisplaying simplices for the valid triangulation:")
        for simplex_var_index in chosen_indices
            simplex_point_indices = S_indices[simplex_var_index]
            display(P[collect(simplex_point_indices), :])
        end
    else
        println("UNSATISFIABLE: No solution was found. Status: $solution")
        if id == 3 # Special check for the known non-triangulable Reeve tetrahedron
            println("This UNSAT result is EXPECTED for the Reeve tetrahedron.")
        end
    end

    push!(timings, "Total execution time" => (time_ns() - t_start_total) / 1e9)
    
    println("\n--- Timing & Memory Summary for Polytope #$id ---")
    peak_ram_bytes = Sys.maxrss()
    for (operation, duration) in timings
        println(@sprintf("%-45s: %.4f seconds", operation, duration))
    end
    println(@sprintf("%-45s: %.2f MiB", "Peak memory usage (Max RSS)", peak_ram_bytes / 1024^2))
end


# --- Main Execution Block ---

function main()
    if isempty(ARGS)
        println("Usage: julia -t auto $(@__FILE__) <range>")
        println("Example: julia -t auto $(@__FILE__) 3")
        println("         julia -t auto $(@__FILE__) 1-2")
        return
    end

    filepath = "polytopes.txt"
    if !isfile(filepath)
        println("Error: Test case file not found at '$filepath'")
        println("Please create this file with the specified test cases.")
        return
    end

    polytopes = read_polytopes_from_file(filepath)
    process_range = parse_range(ARGS[1])

    if process_range.start < 1 || process_range.stop > length(polytopes)
        println("Error: Range is out of bounds. Found $(length(polytopes)) polytopes in the file.")
        return
    end

    for i in process_range
        process_polytope(polytopes[i], i)
    end
end

main()

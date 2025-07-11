using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf

"""
    findAllLatticePointsInHull(vertices::Matrix{Int})

Takes a matrix of vertices defining a convex polytope and returns a matrix
containing all integer lattice points inside its convex hull.
"""
function findAllLatticePointsInHull(vertices::Matrix{Int})
    println("Finding all lattice points inside the convex hull of the input vertices...")
    poly = polyhedron(vrep(vertices))
    hr = hrep(poly)
    min_coords = floor.(Int, minimum(vertices, dims=1))
    max_coords = ceil.(Int, maximum(vertices, dims=1))
    lattice_points = Vector{Vector{Int}}()
    tol = 1e-8

    for iz in min_coords[3]:max_coords[3]
        for iy in min_coords[2]:max_coords[2]
            for ix in min_coords[1]:max_coords[1]
                point = [ix, iy, iz]
                if all(hr.A * point .<= hr.b .+ tol)
                    push!(lattice_points, point)
                end
            end
        end
    end
    
    return vcat(lattice_points'...)
end


"""
    all_unimodular_simplices_in(P)

Finds all 4-tuples of point *indices* in P that form a unimodular simplex.
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

Calculates the convex hull of P once and finds all triangular faces that are
NOT on the boundary.
"""
function precompute_open_faces(P::Matrix{Int}; tol=1e-8)
    n = size(P, 1)
    poly = polyhedron(vrep(P))
    hr = hrep(poly)
    planes = collect(halfspaces(hr))
    open_faces_set = Set{NTuple{3, Int}}()

    for face_indices in combinations(1:n, 3)
        face_points = P[face_indices, :]
        on_boundary = false
        for plane in planes
            if all(abs.(face_points * plane.a .- plane.β) .<= tol)
                on_boundary = true
                break
            end
        end
        if !on_boundary
            push!(open_faces_set, Tuple(sort(collect(face_indices))))
        end
    end
    return open_faces_set
end

# The geometry helper functions are unchanged
function tetrahedra_intersect_volume(tetra1_verts::Matrix, tetra2_verts::Matrix)
    t1 = Float64.(tetra1_verts); t2 = Float64.(tetra2_verts)
    vol1 = abs(dot(t1[1,:] - t1[4,:], cross(t1[2,:] - t1[4,:], t1[3,:] - t1[4,:]))) / 6.0
    vol2 = abs(dot(t2[1,:] - t2[4,:], cross(t2[2,:] - t2[4,:], t2[3,:] - t2[4,:]))) / 6.0
    if vol1 < 1e-9 || vol2 < 1e-9; return false; end
    axes = []; append!(axes, _get_outward_face_normals(t1)); append!(axes, _get_outward_face_normals(t2))
    edges1 = [t1[j,:] - t1[i,:] for (i, j) in combinations(1:4, 2)]
    edges2 = [t2[j,:] - t2[i,:] for (i, j) in combinations(1:4, 2)]
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
        p_fourth = vertices[first(setdiff(1:4, face_indices)), :]
        if dot(normal, p_fourth - p0) > 0; normal = -normal; end
        if norm(normal) > 1e-9; push!(normals, normal); end
    end
    return normals
end


"""
    main()

Main function to set up the problem, generate clauses, and find a triangulation.
"""
function main()
    # Use a Vector of Pairs to preserve the order of timing results.
    timings = Vector{Pair{String, Float64}}()
    t_start_total = time_ns()

    a, b, c = 2, 2, 1
    initial_vertices = [
        0 0 0; a 0 0; 0 b 0; 0 0 c;
        a b 0; a 0 c; 0 b c; a b c
    ]
    
    println("Initial vertices provided:"); display(initial_vertices)

    t_start = time_ns()
    P = findAllLatticePointsInHull(initial_vertices)
    push!(timings, "Find all lattice points" => (time_ns() - t_start) / 1e9)
    println("Found $(size(P, 1)) lattice points in the hull.\n")
    
    t_start = time_ns()
    S_indices = all_unimodular_simplices_in(P)
    push!(timings, "Find all unimodular simplices" => (time_ns() - t_start) / 1e9)
    println("Number of unimodular simplices found: $(length(S_indices))\n")
    
    t_start = time_ns()
    open_faces_set = precompute_open_faces(P)
    push!(timings, "Precompute open faces" => (time_ns() - t_start) / 1e9)
    println("Found $(length(open_faces_set)) unique open faces.\n")
    
    num_simplices = length(S_indices)
    cnf = Vector{Vector{Int}}()
    push!(cnf, collect(1:num_simplices))

    # --- Generate Intersection Clauses ---
    println("[$(Dates.format(now(), "HH:MM:SS"))] Generating intersection clauses...")
    t_start = time_ns()
    num_intersection_clauses = 0
    for i1 in 1:num_simplices, i2 in (i1+1):num_simplices
        s1_verts = P[collect(S_indices[i1]), :]
        s2_verts = P[collect(S_indices[i2]), :]
        if tetrahedra_intersect_volume(s1_verts, s2_verts)
            push!(cnf, [-(i1), -(i2)])
            num_intersection_clauses += 1
        end
    end
    push!(timings, "Generate intersection clauses" => (time_ns() - t_start) / 1e9)
    println("Number of intersection clauses: $num_intersection_clauses")
    
    # --- Generate Face-Covering Clauses ---
    println("[$(Dates.format(now(), "HH:MM:SS"))] Generating face-covering clauses...")
    t_start = time_ns()
    num_face_clauses = 0
    for (i, s_verts_indices) in enumerate(S_indices)
        for face_indices in combinations(s_verts_indices, 3)
            canonical_face = Tuple(sort(collect(face_indices)))
            if canonical_face in open_faces_set
                coverers = [j for (j, s2_verts_indices) in enumerate(S_indices) if i != j && issubset(canonical_face, s2_verts_indices)]
                push!(cnf, vcat([-i], coverers))
                num_face_clauses += 1
            end
        end
    end
    push!(timings, "Generate face-covering clauses" => (time_ns() - t_start) / 1e9)
    println("Number of face-covering clauses: $num_face_clauses")
    println("Total number of clauses: $(length(cnf))\n")

    # --- Solve with PicoSAT ---
    println("[$(Dates.format(now(), "HH:MM:SS"))] Start solving with PicoSAT...\n")
    t_start = time_ns()
    solution = PicoSAT.solve(cnf)
    push!(timings, "Solve SAT problem (PicoSAT)" => (time_ns() - t_start) / 1e9)

    if solution isa Vector{Int}
        println("A satisfying assignment was found!")
        chosen_indices = findall(l -> l > 0, solution)
        expected_volume = 6 * a * b * c
        println("\nNumber of simplices in solution: $(length(chosen_indices))")
        println("Expected number for a full triangulation of the cube: $expected_volume")
        
        @assert length(chosen_indices) == expected_volume "The solution found does not form a valid triangulation of the cube."
        println("Assertion successful: The solution size matches the expected volume for a triangulation.")
        
        println("\nDisplaying simplices for the valid triangulation:")
        for simplex_var_index in chosen_indices
            simplex_point_indices = S_indices[simplex_var_index]
            display(P[collect(simplex_point_indices), :])
        end
    else
        println("UNSATISFIABLE: No solution was found. Status: $solution")
    end

    push!(timings, "Total execution time" => (time_ns() - t_start_total) / 1e9)
    
    println("\n--- Timing Summary ---")
    for (operation, duration) in timings
        println(@sprintf("%-35s: %.4f seconds", operation, duration))
    end
end

main()

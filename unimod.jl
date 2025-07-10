using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT # Use the dedicated PicoSAT solver

"""
    all_unimodular_simplices_in(P)

Finds all 4-tuples of points in P that form a unimodular simplex.
"""
function all_unimodular_simplices_in(P::Matrix{Int})
    n = size(P, 1)
    simplices = []

    for inds in combinations(1:n, 4)
        p0, p1, p2, p3 = P[inds[1], :], P[inds[2], :], P[inds[3], :], P[inds[4], :]
        M = vcat((p1 - p0)', (p2 - p0)', (p3 - p0)')
        d = round(Int, det(M))

        if abs(d) == 1
            push!(simplices, vcat(p0', p1', p2', p3'))
        end
    end
    return simplices
end

"""
    open_faces(s, P, tol=1e-8)

Identifies faces of a simplex `s` that are not on the boundary of the convex hull of `P`.
"""
function open_faces(s::Matrix{Int}, P::Matrix{Int}; tol=1e-8)
    poly = polyhedron(vrep(P))
    hr = hrep(poly)
    planes = collect(halfspaces(hr))
    open_list = []

    for face_indices in combinations(1:4, 3)
        face_points = s[face_indices, :]
        on_boundary = false
        for plane in planes
            a = plane.a
            beta = plane.β
            
            vals = face_points * a .- beta
            if all(abs.(vals) .<= tol)
                on_boundary = true
                break
            end
        end
        if !on_boundary
            push!(open_list, face_points)
        end
    end
    return open_list
end


"""
    tetrahedra_intersect_volume(tetra1_verts, tetra2_verts)

Determines if two tetrahedra have a volumetric intersection using the Separating Axis Theorem (SAT).
"""
function tetrahedra_intersect_volume(tetra1_verts::Matrix, tetra2_verts::Matrix)
    t1 = Float64.(tetra1_verts)
    t2 = Float64.(tetra2_verts)

    # Degeneracy Check
    vol1 = abs(dot(t1[1,:] - t1[4,:], cross(t1[2,:] - t1[4,:], t1[3,:] - t1[4,:]))) / 6.0
    vol2 = abs(dot(t2[1,:] - t2[4,:], cross(t2[2,:] - t2[4,:], t2[3,:] - t2[4,:]))) / 6.0

    if vol1 < 1e-9 || vol2 < 1e-9
        return false
    end

    # Generate potential separating axes
    axes = []
    # Axes from face normals of tetrahedron 1
    append!(axes, _get_outward_face_normals(t1))
    # Axes from face normals of tetrahedron 2
    append!(axes, _get_outward_face_normals(t2))

    # Axes from cross products of edges
    edges1 = [t1[j,:] - t1[i,:] for (i, j) in combinations(1:4, 2)]
    edges2 = [t2[j,:] - t2[i,:] for (i, j) in combinations(1:4, 2)]
    for e1 in edges1
        for e2 in edges2
            axis = cross(e1, e2)
            if norm(axis) > 1e-9
                push!(axes, axis)
            end
        end
    end

    # Perform the Separating Axis Test
    for axis in axes
        min1, max1 = _project(t1, axis)
        min2, max2 = _project(t2, axis)

        if max1 <= min2 || max2 <= min1
            return false
        end
    end

    return true
end

# Helper functions _project and _get_outward_face_normals are unchanged and assumed to be here.
function _project(vertices::Matrix, axis::Vector)
    projections = vertices * axis
    return minimum(projections), maximum(projections)
end

function _get_outward_face_normals(vertices::Matrix)
    normals = []
    face_indices_list = collect(combinations(1:4, 3))

    for face_indices in face_indices_list
        p0, p1, p2 = vertices[face_indices[1], :], vertices[face_indices[2], :], vertices[face_indices[3], :]
        normal = cross(p1 - p0, p2 - p0)
        fourth_vertex_index = first(setdiff(1:4, face_indices))
        p_fourth = vertices[fourth_vertex_index, :]
        if dot(normal, p_fourth - p0) > 0
            normal = -normal
        end
        if norm(normal) > 1e-9
            push!(normals, normal)
        end
    end
    return normals
end


"""
    main()

Main function to set up the problem, generate clauses, and find a triangulation.
"""
function main()
    a, b, c = 2, 2, 2 # Dimensions of the cube for testing

    # Generate points in the cube
    points_vec = [[x, y, z] for x in 0:a for y in 0:b for z in 0:c]
    P = vcat(points_vec'...)

    println("Checking the following lattice polytope:")
    display(P)
    println("P contains $(size(P, 1)) lattice points\n")

    S = all_unimodular_simplices_in(P)
    println("Number of unimodular simplices found: $(length(S))")

    num_simplices = length(S)
    cnf = Vector{Vector{Int}}()

    push!(cnf, collect(1:num_simplices))

    num_intersection_clauses = 0
    for i1 in 1:num_simplices, i2 in (i1+1):num_simplices
        if tetrahedra_intersect_volume(S[i1], S[i2])
            push!(cnf, [-(i1), -(i2)])
            num_intersection_clauses += 1
        end
    end
    println("Number of intersection clauses: $num_intersection_clauses")
    
    num_face_clauses = 0
    for (i, s) in enumerate(S)
        for face in open_faces(s, P)
            coverers = [j for (j, s2) in enumerate(S) if i != j && all(p -> any(q -> q == p, eachrow(s2)), eachrow(face))]
            push!(cnf, vcat([-i], coverers))
            num_face_clauses += 1
        end
    end
    println("Number of face-covering clauses: $num_face_clauses")
    println("Total number of clauses: $(length(cnf))\n")
    println("Start solving with PicoSAT...\n")

    ### PicoSAT SOLVER SECTION ###

    # 1. Solve the problem by passing the CNF list directly to PicoSAT.solve
    solution = PicoSAT.solve(cnf)

    # 2. Check the result
    if solution isa Vector{Int}
        println("✅ A satisfying assignment was found!")
        
        # The solution vector contains the literals that are true.
        # We find the positive literals to identify chosen simplices.
        chosen_indices = findall(l -> l > 0, solution)

        # Post-solution check to see if it matches the volume of the test cube.
        expected_volume = 6 * a * b * c
        println("\nNumber of simplices in solution: $(length(chosen_indices))")
        println("Expected number for a full triangulation of the cube: $expected_volume")
        
        @assert length(chosen_indices) == expected_volume "The solution found does not form a valid triangulation of the cube."
        println("Assertion successful: The solution size matches the expected volume for a triangulation! 🥳")
        
        println("\nDisplaying simplices for the valid triangulation:")
        for idx in chosen_indices
            display(S[idx])
        end
    else
        println("❌ UNSATISFIABLE: No solution was found. Status: $solution")
    end
end

# Run the main function
main()

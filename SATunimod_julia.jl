# SATunimod.jl - Julia translation of the Python script

using Combinatorics
using LinearAlgebra
using Polyhedra # Switched from MiniQhull to Polyhedra
using LogicCircuits

"""
    all_unimodular_simplices_in(P)

Finds all 4-tuples of points in P that form a unimodular simplex.
A simplex is unimodular if the volume is 1/6, which is equivalent to the
determinant of the matrix formed by its edge vectors being ±1.
"""
function all_unimodular_simplices_in(P::Matrix{Int})
    n = size(P, 1)
    simplices = []

    for inds in combinations(1:n, 4)
        p0, p1, p2, p3 = P[inds[1], :], P[inds[2], :], P[inds[3], :], P[inds[4], :]
        M = vcat((p1 - p0)', (p2 - p0)', (p3 - p0)')
        d = round(Int, det(M))

        if abs(d) == 1
            # Store simplex as a 4x3 matrix
            push!(simplices, vcat(p0', p1', p2', p3'))
        end
    end
    return simplices
end

"""
    open_faces(s, P, tol=1e-8)

Identifies faces of a simplex `s` that are not on the boundary of the convex hull of `P`.
These are the "internal" faces that need to be covered by other simplices in a triangulation.
"""
function open_faces(s::Matrix{Int}, P::Matrix{Int}; tol=1e-8)
    # Create a polyhedron from the points using Polyhedra.jl
    poly = polyhedron(vrep(P))
    
    # Get the half-space representation (the boundary planes)
    hr = hrep(poly)
    planes = collect(halfspaces(hr))
    open_list = []

    for face_indices in combinations(1:4, 3)
        face_points = s[face_indices, :]
        on_boundary = false
        for plane in planes
            # A plane is defined by a'x <= β
            # To check if points are on the plane, we check if a'x - β ≈ 0
            a = plane.a
            beta = plane.β
            
            vals = face_points * a .- beta
            if all(abs.(vals) .<= tol)
                on_boundary = true
                break
            end
        end
        if !on_boundary
            # Store face as a 3x3 matrix
            push!(open_list, face_points)
        end
    end
    return open_list
end

"""
    _project(vertices, axis)

Helper function for SAT: Projects vertices onto an axis and returns the min/max interval.
"""
function _project(vertices::Matrix, axis::Vector)
    projections = vertices * axis
    return minimum(projections), maximum(projections)
end

"""
    _get_outward_face_normals(vertices)

Helper function for SAT: Computes the 4 outward-pointing face normals of a tetrahedron.
"""
function _get_outward_face_normals(vertices::Matrix)
    normals = []
    face_indices_list = collect(combinations(1:4, 3))

    for face_indices in face_indices_list
        p0, p1, p2 = vertices[face_indices[1], :], vertices[face_indices[2], :], vertices[face_indices[3], :]
        
        normal = cross(p1 - p0, p2 - p0)
        
        fourth_vertex_index = first(setdiff(1:4, face_indices))
        p_fourth = vertices[fourth_vertex_index, :]
        
        # Ensure the normal points outwards
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
    tetrahedra_intersect_volume(tetra1_verts, tetra2_verts)

Determines if two tetrahedra have a volumetric intersection using the Separating Axis Theorem (SAT).
"""
function tetrahedra_intersect_volume(tetra1_verts::Matrix, tetra2_verts::Matrix)
    t1 = Float64.(tetra1_verts)
    t2 = Float64.(tetra2_verts)

    # Degeneracy Check: Ensure tetrahedra are not flat
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
            return false # Found a separating axis, no volumetric intersection
        end
    end

    return true # No separating axis found, tetrahedra must intersect
end

"""
    main()

Main function to set up the problem, generate clauses, and find a triangulation.
"""
function main()
    a, b, c = 2, 2, 1 # Dimensions of the cube for testing

    # Generate points in the cube
    points_vec = [[x, y, z] for x in 0:a for y in 0:b for z in 0:c]
    P = vcat(points_vec'...) # Convert to a Nx3 matrix

    println("Checking the following lattice polytope:")
    display(P)
    println("P contains $(size(P, 1)) lattice points\n")

    S = all_unimodular_simplices_in(P)
    println("Number of unimodular simplices found: $(length(S))")

    num_simplices = length(S)
    # The CNF instance is a vector of vectors of integers (clauses)
    cnf = Vector{Vector{Int}}()

    # Clause 1: At least one simplex must be chosen.
    # A literal `i` means simplex `i` is chosen, `-i` means it is not.
    push!(cnf, collect(1:num_simplices))

    # Clause 2: Intersection clauses.
    # If two simplices s1 and s2 intersect, we add the clause (-i1 V -i2),
    # meaning we can't have both s1 and s2.
    for (i1, s1) in enumerate(S), (i2, s2) in enumerate(S)
        if i1 < i2 && tetrahedra_intersect_volume(s1, s2)
            push!(cnf, [-(i1), -(i2)])
        end
    end
    println("Number of intersection clauses: $(length(cnf) - 1)")
    
    temp = length(cnf)
    # Clause 3: Face-covering clauses.
    # For each internal face of a simplex `s`, at least one other simplex must cover it.
    # Clause: -s V c1 V c2 V ... (where c_i are coverers)
    for (i, s) in enumerate(S)
        for face in open_faces(s, P)
            coverers = Int[]
            for (j, s2) in enumerate(S)
                if i != j
                    # Check if the face (3 points) is a subset of the vertices of s2
                    is_subset = all(p -> any(q -> q == p, eachrow(s2)), eachrow(face))
                    if is_subset
                        push!(coverers, j)
                    end
                end
            end
            push!(cnf, vcat([-i], coverers))
        end
    end
    println("Number of face-covering clauses: $(length(cnf) - temp)")
    println("Total number of clauses: $(length(cnf))\n")
    println("Start solving now...\n")

    # Create a SAT context and add clauses
    sat_problem = SAT(num_simplices)
    for clause in cnf
        add_clause(sat_problem, clause)
    end

    # Iterate through solutions
    i = 0
    while next_solution(sat_problem)
        sol = get_solution(sat_problem)
        if i % 100 == 0
            println("$i: $(count(x -> x > 0, sol)) - $(6*a*b*c)")
        end
        
        # Check if the number of simplices matches the expected volume
        if count(x -> x > 0, sol) == 6 * a * b * c
            println("Found a triangulation using the following simplices:")
            for (idx, s) in enumerate(S)
                if sol[idx] > 0
                    display(s)
                end
            end
            break # Stop after finding the first valid triangulation
        end
        i += 1
    end
end

# Run the main function
main()

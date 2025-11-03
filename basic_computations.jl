module BasicComputations

using Normaliz
using Combinatorics
using LinearAlgebra
using Polyhedra
using Base.Threads

export lattice_points_via_Normaliz, all_simplices, internal_faces


function lattice_points_via_Normaliz(vertices::Matrix{Int})
    nverts, d = size(vertices)

    # Lift vertices to d+1 dimension by adding 1
    lifted = hcat(vertices, ones(Int, nverts))

    nmz_vertices = Normaliz.NmzMatrix{Normaliz.NmzRational}(lifted)

    # Construct the cone
    cone = Normaliz.LongLongCone(Dict(:cone => nmz_vertices))

    # Get Hilbert basis (generates all integer points in the cone)
    HB = Normaliz.get_matrix_cone_property(cone, "HilbertBasis")

    # Dehomogenize
    ncols = size(HB,2) - 1
    points = []
    for i in 1:size(HB,1)
        len = size(HB,2)
        if HB[i, len] == 1
            push!(points, [HB[i, j] for j in 1:ncols])
        end
    end

    return [BigInt(vec[j]) for vec in points, j in 1:ncols]
end

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

end

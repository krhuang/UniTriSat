module BasicComputations

using Normaliz
using Combinatorics
using LinearAlgebra
using Polyhedra
using Base.Threads
using StaticArrays

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

function next_combination!(inds::Vector{Int}, n::Int)
    k = length(inds)
    for i in k:-1:1
        if inds[i] < n - (k - i)
            inds[i] += 1
            for j in i+1:k
                inds[j] = inds[j-1] + 1
            end
            return true
        end
    end
    return false  # no next combination
end

function all_simplices(lattice_points::Matrix{BigInt}; only_unimodular::Bool=false)
    n, d = size(lattice_points)
    simplex_indices = NTuple{d+1,Int}[]
    if n < d + 1
        return simplex_indices
    end

    inds = collect(1:(d+1))             # initial combination
    diffs = Matrix{BigInt}(undef, d, d)

    while true
        idx_1 = inds[1]
        for j in 1:d
            idx_j = inds[j+1]
            for i in 1:d
                diffs[j, i] = lattice_points[idx_j, i] - lattice_points[idx_1, i]
            end
        end
        det_val = det(diffs)
        if det_val != 0 && (!only_unimodular || abs(det_val) == 1)
            push!(simplex_indices, Tuple(inds))
        end
        next_combination!(inds, n) || break
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

    # Preallocate combination buffer
    inds = collect(1:dim)
    faces = Set{NTuple{dim, Int}}()
    int_plane_a = MVector{dim, Int64}(undef)

    while true
        on_boundary = false
        for plane in planes
            equal = true
            scale = denominator(plane.β) * foldl(lcm, (denominator(x) for x in plane.a))
            int_β = scale * plane.β
            @inbounds for k in 1:dim
                int_plane_a[k] = Int64(scale * plane.a[k])
            end
            @inbounds for j in 1:dim
                s = 0
                ind_j = inds[j]
                for k in 1:dim
                    s += vertices[ind_j, k] * int_plane_a[k]
                end
                if s != int_β
                    equal = false
                    break
                end
            end
            if equal
                on_boundary = true
                break
            end
        end
        if !on_boundary
            push!(faces, Tuple(inds))
        end
        next_combination!(inds, n) || break
    end
    return faces
end

end

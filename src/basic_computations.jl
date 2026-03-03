module BasicComputations

using Combinatorics
using LinearAlgebra
using Polyhedra
using Base.Threads
using StaticArrays
using CDDLib
using AbstractAlgebra

export all_simplices, internal_faces, lattice_points_via_CDDLib, full_dimensional_lattice_projection

const CDD_LIB_EXACT = CDDLib.Library(:exact)


# Given vertices of a lattice polytope, this computes a lattice-preserving projection.
# Here lattice-preserving means it is a lattice-bijection from the lattice points of the affine hull
# The resulting points are the vertices of a full-dimensional lattice polytope
function full_dimensional_lattice_projection(vertices::Matrix{Int})
    # Shifted so that the origin is a vertex
    shifted_vertices = [vertices[i,j] - vertices[1, j] for i in 1:size(vertices,1), j in 1:size(vertices,2)]
    # Convert to a ZZ-matrix, so that we can access AbstractAlgebra's hnf function
    shifted_vertices = matrix(ZZ, shifted_vertices)
    # Compute the HNF of the transpose
    hermite_normal_form = hnf(transpose(shifted_vertices))
    # Record the indices which are non-zero
    non_zero_row_indices = [row_index for row_index in 1:nrows(hermite_normal_form) if !is_zero(hermite_normal_form[row_index,:])]
    # Remove excess all-zero rows and convert to Int64
    new_vertices_transposed = [Int64(hermite_normal_form[i,j]) for i in non_zero_row_indices, j in 1:size(hermite_normal_form,2)]

    # Transpose it back. Due to Julia's standards for transposing we have to take a "copy" here. 
    new_vertices = copy(transpose(new_vertices_transposed))

    return new_vertices
end

# Computes the lattice points of a lattice polytope, via CDDLib backend of Polyhedra
# By default CDDLib uses Floats, but you can configure this

# See the documentation at https://juliapolyhedra.github.io/Polyhedra.jl/stable/polyhedron/

#     "CDDLib.Library creates CDDLib.Polyhedron of type either Float64 or Rational{BigInt}. 
# One can choose the first one using CDDLib.Library(:float) and the second one using 
# CDDLib.Library(:exact), by default it is :float."

# function lattice_points_via_CDDLib(vertices::Matrix{Int})
#     # Convert vertices to exact rationals
#     verts = Rational{BigInt}.(vertices)
#
#     # Build exact polyhedron from vertices
#     poly = polyhedron(vrep(verts), CDDLib.Library(:exact))
#
#     # Compute bounding box
#     min_array = [minimum(vertices[:, i]) for i in 1:size(vertices, 2)]
#     max_array = [maximum(vertices[:, i]) for i in 1:size(vertices, 2)]
#     ranges = [min_array[i]:max_array[i] for i in 1:length(min_array)]
#
#     # Enumerate and collect integer points
#     points_list = Vector{Vector{Int}}()
#
#     for pt_tuple in Iterators.product(ranges...)
#         pt = Rational{BigInt}.(collect(pt_tuple))
#         if in(pt, poly)
#             push!(points_list, collect(Int.(pt_tuple)))
#         end
#     end
#
#     # Convert to a matrix with one point per row
#     if isempty(points_list)
#         return zeros(Int, 0, size(vertices, 2))
#     else
#         return reduce(vcat, [permutedims(p) for p in points_list])
#     end
# end

function lattice_points_via_CDDLib(vertices::Matrix{Int})

    # TODO: write a comment here about floats??
    float_threshold = 1e-10

    verts = Rational{BigInt}.(vertices)
    poly = polyhedron(vrep(verts), CDD_LIB_EXACT)
    hrep_poly = hrep(poly)
    all_halfspaces = halfspaces(hrep_poly)
    num_hyperplanes = length(all_halfspaces)

    if num_hyperplanes == 0
        @warn "Polytop hat keine Hyperebenen-Repräsentation."
        return lattice_points_via_CDDLib(vertices)
    end

    A_rational = reduce(vcat, [h.a' for h in all_halfspaces])
    b_rational = [h.β for h in all_halfspaces]

    A_float = Float64.(A_rational)
    b_float = Float64.(b_rational)

    min_array = [minimum(vertices[:, i]) for i in 1:size(vertices, 2)]
    max_array = [maximum(vertices[:, i]) for i in 1:size(vertices, 2)]
    ranges = [min_array[i]:max_array[i] for i in 1:length(min_array)]

    points_list = Vector{Vector{Int}}()

    for pt_tuple in Iterators.product(ranges...)
        pt_int_vec = collect(Int.(pt_tuple))
        pt_float = Float64.(pt_int_vec)

        use_exact_check = false
        is_outside_float = false

        for i in 1:num_hyperplanes
            dot_prod = dot(@view(A_float[i, :]), pt_float)
            dist = b_float[i] - dot_prod

            if abs(dist) <= float_threshold
                use_exact_check = true
                break
            elseif dist < 0.0
                is_outside_float = true
            end
        end

        if use_exact_check
            pt_rational = Rational{BigInt}.(pt_int_vec)
            if in(pt_rational, poly)
                push!(points_list, pt_int_vec)
            end
        elseif !is_outside_float
            push!(points_list, pt_int_vec)
        end
    end

    if isempty(points_list)
        return zeros(Int, 0, size(vertices, 2))
    else
        return reduce(vcat, [permutedims(p) for p in points_list])
    end
end

#=
# Access the lattice points of a lattice polytope via Normaliz
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

    return [vec[j] for vec in points, j in 1:ncols]
end
=#

# Oscar function for computing lattice points of a convex hull
# Only temporary code...
function lattice_points_via_Oscar(vertices::Matrix{Int})
    polytope = convex_hull(vertices)
    LP = lattice_points(polytope)
    dims = size(LP)
    nrows = dims[1]
    ncols = size(LP[1])[1]
    julia_matrix_LP = [Int64(LP[i][j]) for i in 1:nrows, j in 1:ncols]
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

function all_simplices(lattice_points::Matrix{Int}; unimodular::Bool=true)
    n, d = size(lattice_points)
    simplex_indices = NTuple{d+1,Int}[]
    if n < d + 1
        return simplex_indices
    end

    inds = collect(1:(d+1))             # initial combination
    diffs = Matrix{Int}(undef, d, d)

    while true
        idx_1 = inds[1]
        for j in 1:d
            idx_j = inds[j+1]
            for i in 1:d
                diffs[j, i] = lattice_points[idx_j, i] - lattice_points[idx_1, i]
            end
        end
        det_val = LinearAlgebra.det_bareiss(diffs) # use exact integer determinant
        if det_val != 0 && (!unimodular || abs(det_val) == 1)
            push!(simplex_indices, Tuple(inds))
        end
        next_combination!(inds, n) || break
    end
    return simplex_indices
end

function rational_plane_to_integer(plane)
    denoms = [denominator(x) for x in plane.a]
    lcm_d = foldl(lcm, denoms)
    int_a = map(x -> Int(x * lcm_d), plane.a)
    int_β = Int(plane.β * lcm_d)
    return int_a, int_β
end

function internal_faces(vertices::Matrix{Int}, dim::Int)
    n = size(vertices, 1)
    if n < dim
        return Set{NTuple{dim, Int}}()
    end

    poly = Polyhedra.polyhedron(vrep(vertices), CDD_LIB_EXACT)
    hr = hrep(poly)
    rational_planes = collect(halfspaces(hr))

    # Preallocate combination buffer
    inds = collect(1:dim)
    faces = Set{NTuple{dim, Int}}()

    planes_a = Vector{Vector{Int}}(undef, length(rational_planes))
    planes_β = Vector{Int}(undef, length(rational_planes))
    for i in 1:length(rational_planes)
        planes_a[i], planes_β[i] = rational_plane_to_integer(rational_planes[i])
    end

    while true
        on_boundary = false
        for p in 1:length(planes_a)
            equal = true
            plane_a = planes_a[p]
            plane_β = planes_β[p]
            @inbounds for j in 1:dim
                s = 0
                ind_j = inds[j]
                for k in 1:dim
                    s += vertices[ind_j, k] * plane_a[k]
                end
                if s != plane_β
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

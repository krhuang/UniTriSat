module BasicComputations

using Combinatorics
using LinearAlgebra
using Polyhedra
using Base.Threads
using StaticArrays
using CDDLib
using AbstractAlgebra

using ..Structs

include("Intersection_backends/cpu_intersection.jl")

# mutable flag in module scope
const Normaliz_available = Ref(true)

# Try to import Normaliz.
# If it's not available it gives a small warning and modifies the flag
try
    @eval using Normaliz  # top-level import
    include("Normaliz_backend.jl")
    using .Normaliz_backend
catch e
    Normaliz_available[] = false
end


# mutable flag in module scope
const CUDA_PACKAGES_LOADED = Ref(false)
# try to include Cuda, if its not available and the user wants to use the GPU, then a warning will be printed and we fall back to CPU backend
try
    using CUDA, StaticArrays, CUDA.Adapt
    CUDA_PACKAGES_LOADED[] = true
catch
end
for d in 3:6
    if CUDA_PACKAGES_LOADED[] && isfile("Intersection_backends/gpu_intersection_$(d)d.jl")
        include("Intersection_backends/gpu_intersection_$(d)d.jl")
    end
end

export all_simplices,
    internal_faces, 
    lattice_points_via_CDDLib,
    lattice_points_via_Normaliz,
    lattice_points_via_Oscar,
    compute_lattice_points,
    compute_simplices,
    compute_internal_faces,
    internal_faces,
    compute_intersections_incremental,
    compute_intersections_standard,
    compute_face_clauses,
    find_generic_point,
    compute_central_indices

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


# This function uses floats, but it has a safety exact-check implemented
# So it cannot suffer from floating-point imprecision
function lattice_points_via_CDDLib(vertices::Matrix{Int})

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

# Finds a generic point strictly inside the polytope using Rational{BigInt} arithmetic
function find_generic_point(P::Matrix{Int}, internal_faces_set, ::Val{D}) where D
    n_points_total = size(P, 1)
    max_attempts = 1000

    P_rational = Matrix{Rational{BigInt}}(P)
    face_vectors_f = zeros(MMatrix{D, D - 1, Float64})
    aug_matrix_f = zeros(MMatrix{D, D, Float64})

    for attempt in 1:max_attempts
        weights = rand(1:10000, n_points_total)
        weight_sum = sum(weights)
        p_vec = vec((P_rational' * weights) .// weight_sum)

        is_generic = true

        for face_indices in internal_faces_set
            first_face_index = face_indices[1]
            if length(face_indices) < D
                continue
            end
            @assert length(face_indices) == D
            for j in 1:D-1
                vertex_index = face_indices[j+1]
                for k in 1:D
                    face_vectors_f[k,j] = float(P[vertex_index, k] - P[first_face_index, k])
                end
            end

            r_face = rank(face_vectors_f)
            if r_face < D - 1
                continue
            end

            for j in 1:(D-1)
                aug_matrix_f[:, j] .= face_vectors_f[:, j]
            end
            for k in 1:D
                aug_matrix_f[k, end] = float(p_vec[k] - P[first_face_index, k])
            end
            r_aug = rank(aug_matrix_f)

            if r_face == r_aug
                is_generic = false
                break
            end
        end

        if is_generic
            return p_vec
        end
    end
    error("Could not find a generic point. This should never happen... if you see this error please open an issue on GitHub...")
end

function is_point_in_simplex(P::Matrix{Int}, s_indices, p::Vector{Rational{BigInt}})
    dim = length(p)
    indices = collect(s_indices)
    # The intention is to solve the (d+1)x(d+1) barycentric matrix
    # (with ones in the last row), but we can make things faster by
    # reducing the dimension of the solve by subtracting the first
    # vertex.
    first_vert = Vector{Int}(undef, dim)
    for k in 1:dim
        first_vert[k] = P[s_indices[1], k]
    end
    A = Matrix{Rational{Int}}(undef, dim, dim)
    # Fill A manually: each column = vertex_i - first_vert
    for j in 1:dim
        vert_index = s_indices[j + 1]
        for k in 1:dim
            A[k, j] = P[vert_index, k] - first_vert[k]
        end
    end
    mu = A \ (p - first_vert)
    lambda_last = 1 - sum(mu)
    return all(mu .> 0) && lambda_last > 0
end

function compute_central_indices(P::Matrix{Int}, S_indices, generic_point::Vector{Rational{BigInt}})
    central_indices_map = Int[]
    for (i, s) in enumerate(S_indices)
        if is_point_in_simplex(P, s, generic_point)
            push!(central_indices_map, i)
        end
    end
    return central_indices_map
end

# Computes all lattice points via Normaliz or CDDLib
function compute_lattice_points(initial_vertices::Matrix{Int}, config::Config)
    if Normaliz_available[] && config.use_normaliz
        return lattice_points_via_Normaliz(initial_vertices)
    else
        return lattice_points_via_CDDLib(initial_vertices)
    end
end

# Generates all possible simplices
function compute_simplices(P::Matrix{Int}, config::Config)
    return all_simplices(P, unimodular=config.unimodular)
end

# Generates internal faces
function compute_internal_faces(P::Matrix{Int}, dim::Int)
    return internal_faces(P, dim)
end

# Helper for incremental intersection logic
function compute_intersections_incremental(P::Matrix{Int}, S_indices, internal_faces_set, dim::Int, num_lattice_points::Int)
    local_clauses = Vector{Vector{Int}}()

    # 4a. Find Generic Point
    generic_point = find_generic_point(P, internal_faces_set, Val(dim))
    
    # 4b. Identify Central Simplices & Compute Full Intersections for them
    central_indices_map = compute_central_indices(P, S_indices, generic_point)

    if !isempty(central_indices_map)
        central_S_indices = S_indices[central_indices_map]
        # all simplices containing the generic point intersect with each other
        central_clauses = [[-i, -j] for i in 1:length(central_S_indices) for j in (i+1):length(central_S_indices)]
        for c in central_clauses
            mapped_clause = [x < 0 ? -central_indices_map[abs(x)] : central_indices_map[abs(x)] for x in c]
            push!(local_clauses, mapped_clause)
        end
    end

    # 4c. Hyperplane Separation Logic
    S_idx_map = Dict(Tuple(sort(collect(s))) => i for (i,s) in enumerate(S_indices))
    for face_indices_iter in combinations(1:num_lattice_points, dim)
        face_indices = collect(face_indices_iter)
        face_verts = [P[i, :] for i in face_indices]
        normal = CPUIntersection.compute_face_normal(face_verts, Val(dim))
        if all(iszero, normal); continue; end

        left_simplices = Int[]
        right_simplices = Int[]
        v_ref = P[face_indices[1], :]

        for p_idx in 1:num_lattice_points
            if p_idx in face_indices; continue; end
            candidate_s = copy(face_indices)
            push!(candidate_s, p_idx)
            sort!(candidate_s)
            candidate_tuple = Tuple(candidate_s)

            if haskey(S_idx_map, candidate_tuple)
                s_global_idx = S_idx_map[candidate_tuple]
                val = 0
                p_coords = P[p_idx, :]
                for k in 1:dim
                    val += normal[k] * (p_coords[k] - v_ref[k])
                end
                if val > 0
                    push!(left_simplices, s_global_idx)
                elseif val < 0
                    push!(right_simplices, s_global_idx)
                end
            end
        end

        for i in 1:length(left_simplices)
            s1 = left_simplices[i]
            for j in (i+1):length(left_simplices)
                s2 = left_simplices[j]
                push!(local_clauses, [-s1, -s2])
            end
        end
        for i in 1:length(right_simplices)
            s1 = right_simplices[i]
            for j in (i+1):length(right_simplices)
                s2 = right_simplices[j]
                push!(local_clauses, [-s1, -s2])
            end
        end
    end
    return unique(local_clauses)
end

# Helper for standard (potentially GPU) intersection logic
function compute_intersections_standard(P::Matrix{Int}, S_indices, dim::Int, config::Config, log_verbose::Function)
    intersect_func = nothing
    use_gpu = false

    # load the right GPU backend if required, or fall back to CPU
    if config.intersection_backend == "gpu"
        if dim == 3 && isdefined(@__MODULE__, :GPUIntersection3D)
            log_verbose("     Using 3D GPU backend...")
            intersect_func = () -> GPUIntersection3D.get_intersecting_pairs_gpu(P, S_indices)
            use_gpu = true
        elseif dim == 4 && isdefined(@__MODULE__, :GPUIntersection4D)
            log_verbose("     Using 4D GPU backend...")
            intersect_func = () -> GPUIntersection4D.get_intersecting_pairs_gpu_4d(P, S_indices)
            use_gpu = true
        elseif dim == 5 && isdefined(@__MODULE__, :GPUIntersection5D)
            log_verbose("     Using 5D GPU backend...")
            intersect_func = () -> GPUIntersection5D.get_intersecting_pairs_gpu_5d(P, S_indices)
            use_gpu = true
        elseif dim == 6 && isdefined(@__MODULE__, :GPUIntersection6D)
            log_verbose("     Using 6D GPU backend...")
            intersect_func = () -> GPUIntersection6D.get_intersecting_pairs_gpu_6d(P, S_indices)
            use_gpu = true
        end
    end
    
    if use_gpu && !isnothing(intersect_func)
        return intersect_func() # Execute the selected GPU function
    else
        if config.intersection_backend == "gpu"
            log_verbose("     WARNING: GPU backend for $(dim)D not available. Falling back to CPU.")
        end
        log_verbose("     Using CPU backend.")
        return CPUIntersection.get_intersecting_pairs_cpu_generic(P, S_indices, Val(dim))
    end
end

# Generates face-covering clauses
function compute_face_clauses(S_indices, internal_faces_set, dim::Int)
    n_simplices = length(S_indices)
    next_simplex_idx = Threads.Atomic{Int}(1)
    
    tasks = [
        Threads.@spawn begin
            local_clauses = Vector{Vector{Int}}()
            while true
                i = Threads.atomic_add!(next_simplex_idx, 1)
                if i > n_simplices; break; end
                for face_indices in combinations(S_indices[i], dim)
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
    return vcat(fetch.(tasks)...)
end

end

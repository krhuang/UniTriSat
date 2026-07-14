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
    CPUIntersection,
    internal_faces, 
    lattice_points_via_CDDLib,
    lattice_points_via_Normaliz,
    # lattice_points_via_Oscar,
    compute_lattice_points,
    compute_simplices,
    compute_internal_faces,
    internal_faces,
    compute_intersections_incremental,
    compute_intersections_standard,
    compute_face_clauses,
    find_generic_point,
    compute_central_indices,
    full_dimensional_lattice_projection


const CDD_LIB_EXACT = CDDLib.Library(:exact)


# ====================================================================
# HELPER: Full Dimensional Lattice Projection via HNF
# ====================================================================
function full_dimensional_lattice_projection(initial_vertices::Matrix{Int64})
    # Shift to the origin using the first vertex as an anchor
    v0 = initial_vertices[1, :] # Assuming vertices are rows (N x dim)
    shifted_vertices = initial_vertices .- transpose(v0)
    
    # Convert to AbstractAlgebra/Nemo integer matrix
    # We want to eliminate dependent ambient dimensions (columns).
    # HNF clears rows, so we transpose: rows become ambient dimensions.
    M = matrix(ZZ, Matrix(transpose(shifted_vertices)))
    
    H = hnf(M) # We don't even need the transform matrix U for just coordinates!
    
    # Identify non-zero coordinate rows in the HNF matrix
    # The zero-rows at the bottom represent the redundant ambient dimensions.
    nz_row_indices = Int64[]
    for r in 1:nrows(H)
        if !is_zero(H[r, :])
            push!(nz_row_indices, r)
        end
    end
    
    # Extract the valid lower-dimensional lattice coordinates
    # H[nz_row_indices, :] gives us a (new_dim x N) matrix.
    H_clean = [Int64(H[r, c]) for r in nz_row_indices, c in 1:ncols(H)]
    
    # Shift back if needed, and transpose back to (N x new_dim)
    # We must explicitly convert out of the lazy transpose wrapper for CDDLib safety
    projected_vertices = Matrix(transpose(H_clean))
    
    return projected_vertices
end

function lattice_points_via_CDDLib(vertices::Matrix{Int})

    # TODO: write a comment here about floats??
    float_threshold = 1e-10

    verts = Rational{BigInt}.(vertices)
    poly = polyhedron(vrep(verts), CDD_LIB_EXACT)
    hrep_poly = hrep(poly)
    all_halfspaces = halfspaces(hrep_poly)
    num_hyperplanes = length(all_halfspaces)

    if num_hyperplanes == 0
        @warn "Polytope has no H-representation"
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

# function lattice_points_via_Oscar(vertices::Matrix{Int})
#     polytope = convex_hull(vertices)
#     LP = lattice_points(polytope)
#     dims = size(LP)
#     nrows = dims[1]
#     ncols = size(LP[1])[1]
#     julia_matrix_LP = [Int64(LP[i][j]) for i in 1:nrows, j in 1:ncols]
#     return julia_matrix_LP
# end

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
    if n < d + 1
        return NTuple{d+1, Int}[]
    end

    thread_results = [NTuple{d+1, Int}[] for _ in 1:nthreads()]

    @threads for tid in 1:nthreads()
        diffs = Matrix{Int}(undef, d, d)

        for i in tid:nthreads():(n - d)
            pool_start = i + 1
            pool_n = n - i

            if pool_n >= d
                c = collect(1:d)
                while true
                    for r in 1:d
                        p_idx = pool_start + c[r] - 1
                        for col in 1:d
                            diffs[r, col] = lattice_points[p_idx, col] - lattice_points[i, col]
                        end
                    end

                    # Use exact integer arithmetic.
                    det_val = LinearAlgebra.det_bareiss(diffs)

                    if det_val != 0 && (!unimodular || abs(det_val) == 1)
                        simplex = ntuple(j -> j == 1 ? i : pool_start + c[j-1] - 1, d + 1)
                        push!(thread_results[tid], simplex)
                    end

                    k = d
                    while k > 0 && c[k] == pool_n - d + k
                        k -= 1
                    end
                    k == 0 && break

                    c[k] += 1
                    for j in k+1:d
                        c[j] = c[k] + (j - k)
                    end
                end
            end
        end
    end

    return reduce(vcat, thread_results)
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

# Finds a point strictly inside the polytope that lies on no hyperplane spanned by lattice points inside the polytope
function find_generic_point(P::Matrix{Int}, internal_faces_set, ::Val{D}) where D
    n_points_total = size(P, 1)
    max_attempts = 1000

    P_rational = Matrix{Rational{BigInt}}(P)

    # Precompute (integer normal, anchor vertex index) for every internal face
    # that spans a hyperplane; degenerate faces (zero normal) span no
    # hyperplane and impose no genericity constraint.
    hyperplanes = Vector{Tuple{Vector{Int64}, Int}}()
    for face_indices in internal_faces_set
        if length(face_indices) != D
            continue
        end
        face_verts = [Vector{Int}(P[idx, :]) for idx in face_indices]
        normal = CPUIntersection.compute_face_normal(face_verts, Val(D))
        if all(iszero, normal)
            continue
        end
        push!(hyperplanes, (collect(Int64, normal), face_indices[1]))
    end

    for attempt in 1:max_attempts
        weights = rand(1:10000, n_points_total)
        weight_sum = sum(weights)
        p_vec = vec((P_rational' * weights) .// weight_sum)

        is_generic = true
        for (normal, anchor_idx) in hyperplanes
            s = zero(Rational{BigInt})
            for k in 1:D
                s += normal[k] * (p_vec[k] - P[anchor_idx, k])
            end
            if iszero(s)
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
    # Rational{BigInt}: intermediate values in the elimination can overflow
    # machine integers even for moderate coordinates.
    A = Matrix{Rational{BigInt}}(undef, dim, dim)
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

# Builds the reduced ("small") intersection clause set used by incremental
# solving. It consists of
#   (i)  an exactly-one structure over the central simplices, i.e. the
#        simplices whose interior contains a fixed generic point: an OR clause
#        (every triangulation covers the generic point, so it uses at least
#        one central simplex) plus pairwise conflict clauses (any two central
#        simplices overlap in the generic point), and
#   (ii) hyperplane-separation clauses: two simplices sharing the same
#        potential facet F, with their apexes strictly on the same side of
#        aff(F), overlap in a neighbourhood of relint(F) and therefore
#        exclude each other.
# Together with the face-covering clauses, this formula has exactly the
# unimodular triangulations of P as its solutions while being much smaller
# than the full pairwise intersection clause set. The remaining (redundant)
# intersection clauses are streamed to the solver later; see
# solve_cadical_incremental.
function compute_intersections_incremental(P::Matrix{Int}, S_indices, internal_faces_set, dim::Int, num_lattice_points::Int)
    local_clauses = Vector{Vector{Int}}()

    # (i) Generic point and the exactly-one structure over central simplices
    generic_point = find_generic_point(P, internal_faces_set, Val(dim))
    central_indices_map = compute_central_indices(P, S_indices, generic_point)

    if isempty(central_indices_map)
        # Every triangulation must cover the generic point, so this can only
        # happen if something upstream went wrong (e.g. a non-generic point).
        error("No simplex contains the generic point; this should never happen. Please open an issue on GitHub.")
    end

    # OR: some central simplex is used. This subsumes (and strengthens) the
    # global non-emptiness clause.
    push!(local_clauses, copy(central_indices_map))
    # Pairwise conflicts: at most one central simplex is used.
    for a in 1:length(central_indices_map)
        for b in (a+1):length(central_indices_map)
            s1, s2 = minmax(central_indices_map[a], central_indices_map[b])
            push!(local_clauses, [-s1, -s2])
        end
    end

    # (ii) Hyperplane separation, parallelized over the smallest point index
    # of the face. Workers pull the next first-index from an atomic counter
    # and write into thread-local clause lists.
    S_idx_map = Dict(Tuple(sort(collect(s))) => i for (i,s) in enumerate(S_indices))
    next_first_index = Threads.Atomic{Int}(1)

    separation_tasks = map(1:nthreads()) do _
        Threads.@spawn begin
            clauses = Vector{Vector{Int}}()
            left_simplices = Int[]
            right_simplices = Int[]

            while true
                f1 = Threads.atomic_add!(next_first_index, 1)
                if f1 > num_lattice_points - dim + 1
                    break
                end

                for rest in combinations((f1+1):num_lattice_points, dim - 1)
                    face_indices = vcat(f1, rest)
                    face_verts = [P[i, :] for i in face_indices]
                    normal = CPUIntersection.compute_face_normal(face_verts, Val(dim))
                    if all(iszero, normal); continue; end

                    empty!(left_simplices)
                    empty!(right_simplices)
                    v_ref = P[f1, :]

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

                    for side in (left_simplices, right_simplices)
                        for i in 1:length(side)
                            for j in (i+1):length(side)
                                s1, s2 = minmax(side[i], side[j])
                                push!(clauses, [-s1, -s2])
                            end
                        end
                    end
                end
            end

            clauses
        end
    end

    for task in separation_tasks
        append!(local_clauses, fetch(task))
    end

    # The same pair of simplices can conflict via several shared faces (and
    # via the central clauses); clauses are normalized above so unique
    # deduplicates them.
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

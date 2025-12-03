# cpu_intersection.jl

module CPUIntersection

using LinearAlgebra
using Combinatorics
using Base.Threads
using StaticArrays

export get_intersecting_pairs_cpu, simplices_intersect_sat_cpu, compute_face_normal, prepare_simplices_cpu
export check_intersections_range_cpu, find_intersections_in_subset

macro generate_determinants_up_to(n)
    n = Int(n)

    # Generate determinant function for dimension k
    function make_det_func(k)
        fname = Symbol("det", k, "x", k)

        # collect all matrix argument symbols m11, m12, ..., mkk
        args = [Symbol(:m, i, j) for i in 1:k, j in 1:k]

        if k == 2
             return quote
                @inline function $(esc(fname))($(args...))
                    return m11*m22 - m12*m21
                end
            end
        end

        # Laplace expansion along the first row
        subcalls = Expr[]
        for j in 1:k
            subcols = filter(!=(j), 1:k)
            subargs = []
            for i in 2:k
                for c in subcols
                    push!(subargs, Symbol(:m, i, c))
                end
            end
            subcall = Expr(:call, Symbol("det", k-1, "x", k-1), subargs...)
            sign = (-1)^(1 + j)
            push!(subcalls, :($(sign) * $(Symbol(:m, 1, j)) * $subcall))
        end

        quote
            # Combine the terms: m11*d1 - m12*d2 + ...
            @inline function $(esc(fname))($(args...))
                $(foldl((x,y)->:($x + $y), subcalls))
            end
        end
    end

    # Generate all functions det2x2 through det{n}x{n}
    det_funcs = [make_det_func(k) for k in 2:n]

    quote
        $(det_funcs...)
    end
end

@generate_determinants_up_to 5

macro generate_determinants_bareiss_up_to(n)
    n = Int(n)

    function make_det_func(k)
        fname = Symbol("det", k, "x", k, "_bareiss")
        args = [Symbol(:m, i, j) for i in 1:k, j in 1:k]

        stmts = Expr[]

        # Initialize all a_ij
        for i in 1:k, j in 1:k
             push!(stmts, :($(Symbol(:a_, i, "_", j)) = $(Symbol(:m, i, j))))
        end

        # Unroll elimination updates
        for pivot in 1:k-1
            app = Symbol(:a_, pivot, "_", pivot)

            # Do row-swapping to avoid division-by-zero
            if pivot < k - 1
                inner = quote
                    # no non-zero entries found, means the matrix is singular
                    return 0
                end

                for m in reverse((pivot+1):k)
                    swap_code = Expr[]
                    for i in 1:k
                        ami = Symbol(:a_, m, "_", i)
                        api = Symbol(:a_, pivot, "_", i)
                        push!(swap_code, :(tmp = $ami; $ami = $api; $api = tmp))
                    end

                    amp = Symbol(:a_, m, "_", pivot)
                    inner = quote
                        if $amp != 0
                            sign = -sign
                            $(swap_code...)
                        else
                            $inner
                        end
                    end
                end

                # Check if a row swap is needed
                push!(stmts,
                      quote
                          if $app == 0
                              $inner
                          end
                      end)
            end

            for i in (pivot+1):k
                aip = Symbol(:a_, i, "_", pivot)
                for j in (pivot+1):k
                    aij = Symbol(:a_, i, "_", j)
                    apj = Symbol(:a_, pivot, "_", j)
                    denom = Symbol(:a_, pivot-1, "_", pivot-1)
                    update = :($aij * $app - $aip * $apj)
                    update_div = if pivot == 1
                        update
                    else
                        :(Core.Intrinsics.sdiv_int($update, $denom))
                    end
                    push!(stmts, :($aij = $update_div))
                end
            end
        end

        push!(stmts,
              quote
                  return sign * $(Symbol(:a_, k, "_", k))
              end)

        quote
            @inline function $(esc(fname))($(args...))
                sign = 1
                $(stmts...)
            end
        end
    end

    det_funcs = [make_det_func(k) for k in 2:n]

    quote
        $(det_funcs...)
    end
end

@generate_determinants_bareiss_up_to 5

macro generate_gcross_unrolled_full(d)
    d_val = Int(d)
    fname = Symbol("gcross", d_val)

    # Generate variable symbols for each matrix entry
    vsym = [Symbol("v", i, "_", j) for i in 1:d_val, j in 1:(d_val-1)]
    comp_exprs = Expr[]
    rows_full = 1:d_val

    # Generate the generalized cross product using det{d-1}x{d-1}
    for i in 1:d_val
        rows_minor = filter(x -> x != i, rows_full)
        subargs = []
        for r in rows_minor, c in 1:(d_val-1)
            push!(subargs, vsym[r, c])
        end
        # Use the Bareiss algorithm for 5x5 determinants
        det_name = if d_val >= 6
            Symbol("det", d_val-1, "x", d_val-1, "_bareiss")
        else
            Symbol("det", d_val-1, "x", d_val-1)
        end
            call_expr = Expr(:call, det_name, subargs...)
        if isodd(i)
            call_expr = :(-$call_expr)
        end
        push!(comp_exprs, :( $(Symbol("n", i)) = $call_expr ))
    end

    # Build scalar entry function
    scalar_args = vec(vsym)
    scalar_func_name = Symbol(fname, "_scalar!")
    scalar_func = quote
        @inline function $(esc(scalar_func_name))($(scalar_args...))
            $( [:( $(Symbol("n", i)) = 0 ) for i in 1:d_val ]... )
            @inbounds begin
                $(comp_exprs...)
                return SVector{$d_val, Int64}($((Symbol("n", i) for i in 1:d_val)...))
            end
        end
    end

    # Build vec wrapper function
    vec_func = quote
        function $(esc(:_generalized_cross_product))(vs::MVector{$d_val - 1, SVector{$d_val, Int64}})
            # unpack vector of vectors into scalar variables
            $(Expr(:block, [:( $(Symbol("v", i, "_", j)) = vs[$j][$i] ) for i in 1:d_val, j in 1:(d_val-1)]...))
            # call scalar function with all the scalar variables
            return $(esc(scalar_func_name))(
                $( [Symbol("v", i, "_", j) for i in 1:d_val, j in 1:(d_val-1)]... )
            )
        end
    end

    quote
        $scalar_func
        $vec_func
    end
end

@generate_gcross_unrolled_full 3
@generate_gcross_unrolled_full 4
@generate_gcross_unrolled_full 5
@generate_gcross_unrolled_full 6

function _generalized_cross_product(vectors::MVector{N, SVector{D, Int64}}) where {N, D}
    n = N
    d = D
    @assert N == D - 1
    normal = zeros(MVector{D, Int64})
    tmp = Matrix{Int64}(undef, n, n)
    sign = 1

    @inbounds for i in 1:d
        # Computing the minor excluding the ith row
        row_dst = 1
        for row_src in 1:d
            if row_src == i
                continue
            end
            # copy row_src-th vector into tmp[row_dst, :]
            for j in 1:n
                tmp[row_dst, j] = vectors[j][row_src]
            end
            row_dst += 1
        end
        sign = -sign
        normal[i] = sign * LinearAlgebra.det_bareiss(tmp)
    end
    return normal
end

# --- NEW FUNCTION START ---
# Exposes the generalized cross product to the outside world for hyperplane computation
function compute_face_normal(face_verts::Vector{Vector{Int}}, ::Val{D}) where D
    # The generalized cross product takes d-1 vectors (edges of the face)
    # We construct edges relative to the first vertex of the face
    # face_verts has d vertices.
    p0 = SVector{D, Int64}(face_verts[1])
    # Create the MVector of SVectors expected by the internal function
    # Size is D-1 (number of edge vectors)
    span = MVector{D - 1, SVector{D, Int64}}(
        SVector{D, Int64}(face_verts[i] - face_verts[1]) for i in 2:D
    )
    return _generalized_cross_product(span)
end
# --- NEW FUNCTION END ---

struct Simplex{V, D}
    verts::SVector{V, SVector{D, Int64}}         # (num_verts) x d
    facet_normals::SVector{V, SVector{D, Int64}} # list of normals (one per facet)
    face_edges::Vector{Vector{Vector{SVector{D, Int64}}}} # maps face_dim => list of edge vectors (j>i) per face
end

# Evil macro
macro generate_cross_axes_case_scalar(d)
    d_val = Int(d)
    stmts = Expr[]
    scalar_func = Symbol("gcross", d_val, "_scalar!")

    for l in 1:div(d_val - 1, 2)
        k = d_val - 1 - l

        edge_count = binomial(d_val + 1, k + 1)

        inner = quote
            s1_face_edges_k = $(esc(:s1_face_edges))[$k]
            s2_face_edges_l = $(esc(:s2_face_edges))[$l]
            @inbounds for i in 1:$edge_count
                f1_edges = s1_face_edges_k[i]
                # Hoist accesses into local variables
                $( [:( $(Symbol("f1_edge_", j)) = f1_edges[$j] )
                    for j in 1:k ]... )
                $( [:( $(Symbol("v1_", j, "_", m)) = $(Symbol("f1_edge_", j))[$m] )
                     for j in 1:k, m in 1:d_val ]... )
                for j in 1:$edge_count
                    f2_edges = s2_face_edges_l[j]
                    # Hoist accesses into local variables
                    $( [:( $(Symbol("f2_edge_", j)) = f2_edges[$j] )
                        for j in 1:l ]... )
                    $( [:( $(Symbol("v2_", j, "_", m)) = $(Symbol("f2_edge_", j))[$m] )
                        for j in 1:l, m in 1:d_val ]... )

                    # call scalar gcross using local variables
                    axis = $scalar_func(
                        $( [Symbol("v1_", j, "_", m) for j in 1:k, m in 1:d_val ]... ),
                        $( [Symbol("v2_", j, "_", m) for j in 1:l, m in 1:d_val ]... )
                    )

                    if any(!iszero, axis) && axis_separates($(esc(:s1_verts)), $(esc(:s2_verts)), axis)
                        return false
                    end
                end
            end
        end

        push!(stmts, inner)
    end

    return Expr(:block, stmts...)
end

@inline function axis_separates(s1_verts::SVector{V, SVector{D, Int64}},
                                s2_verts::SVector{V, SVector{D, Int64}},
                                axis) where {V, D}
    projs1 = ntuple(i -> dot(s1_verts[i], axis), Val(V))
    projs2 = ntuple(i -> dot(s2_verts[i], axis), Val(V))
    return maximum(projs1) <= minimum(projs2) || maximum(projs2) <= minimum(projs1)
end

function simplices_intersect_sat_cpu(s1::Simplex{V, D}, s2::Simplex{V, D}) where {V, D}
    s1_verts = s1.verts
    s2_verts = s2.verts
    s1_face_edges = s1.face_edges
    s2_face_edges = s2.face_edges

    # facets
    for simplex in (s1, s2)
        facet_normals = simplex.facet_normals
        for i in 1:V
            axis = facet_normals[i]
            if !iszero(axis) && axis_separates(s1_verts, s2_verts, axis)
                 return false
            end
        end
    end

    # cross case
    if D == 3
        @generate_cross_axes_case_scalar 3
    elseif D == 4
        @generate_cross_axes_case_scalar 4
    elseif D == 5
        @generate_cross_axes_case_scalar 5
    elseif D == 6
        @generate_cross_axes_case_scalar 6
    else
        edgeset = zeros(MVector{D - 1, SVector{D, Int64}})
        for k in 1:div((D - 1), 2)
            l = D - 1 - k
            edge_count = binomial(D + 1, k + 1)
            s1_face_edges_k = s1_face_edges[k]
            s2_face_edges_l = s1_face_edges[l]
            for i in 1:edge_count
                f1_edges = s1_face_edges_k[i]
                # combine edges spanning the two faces
                for j in 1:k
                    edgeset[j] = f1_edges[j]
                end
                for j in 1:edge_count
                    f2_edges = s2_face_edges_l[j]
                    for j in 1:l
                        edgeset[k+j] = f2_edges[j]
                    end
                    axis = _generalized_cross_product(edgeset)
                    if !iszero(axis) && axis_separates(s1_verts, s2_verts, axis)
                        return false
                    end
                end
            end
        end
    end

    # We've enumerated and tested all possible axes but none of them
    # separate. Therefore, the simplices must intersect.
    return true
end

# Compute facet normals and edge vectors for a single simplex
function compute_simplex_data(verts::SVector{V, SVector{D, Int64}}) where {V, D}
    num_verts = D + 1

    facet_normals = Vector{SVector{D, Int64}}()

    # Precompute all edges and index map
    num_edges = binomial(num_verts, 2)
    edges = Vector{SVector{D, Int64}}(undef, num_edges)
    edge_index = Vector{Vector{Int64}}(undef, num_verts - 1)
    let edges_idx = 1
        for i in 1:(num_verts-1)
             edge_index[i] = Vector{Int64}(undef, num_verts)
            for j in (i+1):num_verts
                edges[edges_idx] = verts[j] - verts[i]
                edge_index[i][j] = edges_idx
                edges_idx += 1
            end
        end
    end

     # facets are combinations(1:num_verts, d)
    for off_index in 1:num_verts
        first_index = (off_index == 1) ? 2 : 1
        p0 = verts[first_index]
        span = MVector{D - 1, SVector{D, Int64}}(verts[facet_index] - p0
                                                 for facet_index in (first_index + 1):num_verts
                                                 if facet_index != off_index)
        normal = _generalized_cross_product(span)  # Int64 vector length d
        if all(iszero, normal)
            continue
        end
        p_off_face = verts[off_index]
        if dot(normal, p_off_face - p0) > 0
            normal = -normal
        end
        push!(facet_normals, normal)
    end

    # --- precompute faces and exactly k spanning vectors per face ---
    face_edges = Vector{Vector{Vector{SVector{D, Int64}}}}(undef, D - 1)
    for k in 1:(D-1)  # face dimension
        face_edges[k] = Vector{Vector{SVector{D, Int64}}}(undef, binomial(num_verts, k + 1))
        for (i, face_indices) in enumerate(combinations(1:num_verts, k+1))
            # collect exactly k spanning edges for the generalized cross
            spanning_edges = Vector{SVector{D, Int64}}(undef, k)
            p0 = face_indices[1]
            for j in 2:(k+1)
                pj = face_indices[j]
                spanning_edges[j - 1] = edges[p0 < pj ? (edge_index[p0][pj]) : edge_index[pj][p0]]
            end
            face_edges[k][i] = spanning_edges
        end
    end

    return Simplex{D + 1, D}(verts,
                             SVector{D+1, SVector{D, Int64}}(facet_normals),
                             face_edges)
end

# Precompute the type conversion and also the generalized cross
# products for each simplex
function prepare_simplices_cpu(P::Matrix{Int}, S_indices::Vector, ::Val{D}) where D
    num_simplices = length(S_indices)
    simplices = Vector{Simplex{D + 1, D}}(undef, num_simplices)
    for i in 1:num_simplices
        verts = P[collect(S_indices[i]), :]
        num_verts = size(verts, 1)  # should equal d+1 for simplices
        @assert num_verts == D + 1 "expected each simplex to have d+1 vertices"
        sverts = SVector{D + 1, SVector{D,Int64}}(SVector{D,Int64}(verts[i, :]) for i in 1:(D + 1))
        simplices[i] = compute_simplex_data(sverts)
    end
    return simplices
end

function get_intersecting_pairs_cpu_generic(P::Matrix{Int}, S_indices::Vector, ::Val{D}) where D
    simplices::Vector{Simplex{D+1, D}} = prepare_simplices_cpu(P, S_indices, Val(D))
    num_simplices = length(simplices)
    if num_simplices <= 1
        return Vector{Vector{Int}}()
    end

    num_threads = nthreads()
    thread_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()]
    block_size = div(num_simplices + num_threads - 1, num_threads)
    @threads for thread_id in 1:num_threads
        i_start = (thread_id - 1) * block_size + 1
        i_end = min(thread_id * block_size, num_simplices - 1)

        clauses = thread_clauses[thread_id]

        for i in i_start:i_end
            t1 = simplices[i]
            for j in i+1:num_simplices
                t2 = simplices[j]
                if simplices_intersect_sat_cpu(t1, t2)
                    push!(clauses, [-i, -j])
                end
            end
        end
    end

    return vcat(thread_clauses...)
end

# --- NEW HELPERS FOR INCREMENTAL SOLVING ---

# Checks intersections for a single simplex `idx1` against a range of other simplices.
# Efficient for use in worker threads.
function check_intersections_range_cpu(
    simplices::Vector{Simplex{V, D}},
    idx1::Int,
    range_start::Int,
    range_end::Int
) where {V, D}
    conflicts = Vector{Vector{Int}}()
    # If range is invalid or empty, return immediately
    if range_start > range_end
        return conflicts
    end

    t1 = simplices[idx1]
    @inbounds for idx2 in range_start:range_end
        t2 = simplices[idx2]
        if simplices_intersect_sat_cpu(t1, t2)
            push!(conflicts, [-idx1, -idx2])
        end
    end
    return conflicts
end

# Validates a specific subset of simplices (e.g. a candidate solution)
# Returns a list of conflicting pairs found within this subset.
function find_intersections_in_subset(
    simplices::Vector{Simplex{V, D}},
    indices::Vector{Int}
) where {V, D}
    conflicts = Vector{Vector{Int}}()
    n = length(indices)
    for i in 1:n
        idx1 = indices[i]
        t1 = simplices[idx1]
        for j in (i+1):n
            idx2 = indices[j]
            t2 = simplices[idx2]
            if simplices_intersect_sat_cpu(t1, t2)
                push!(conflicts, [-idx1, -idx2])
            end
        end
    end
    return conflicts
end

end
# cpu_intersection.jl
#
#   [FIX-1] gcross argument order was transposed for k >= 2, so every axis built
#           from a face of dimension >= 2 was computed from a garbled matrix.
#   [FIX-2] Simplex is now isbits.  The per-simplex
#           Vector{Vector{Vector{SVector{D,Int64}}}} forest (2^(D+1)-3 heap
#           objects per simplex; 125 at D=6) is gone.  Face->vertex tables are
#           module-level constants shared by every simplex, and spanning vectors
#           are formed on the fly.  Vector{Simplex} is now one contiguous
#           allocation.  This is the memory fix.
#   [FIX-3] Degenerate simplices no longer throw DimensionMismatch: a degenerate
#           facet gets the zero normal, which the SAT loop already skips.
#   [FIX-4] Bareiss zero-pivot guard is emitted for *every* pivot (it used to be
#           skipped for the last one), and Core.Intrinsics.sdiv_int is replaced
#           by div.
#   [FIX-5] AABB prefilter.  Coordinate axes are legitimate candidate axes, so a
#           hit is a genuine separation certificate; it kills most pairs of a
#           triangulation before the ~3300-axis loop at D=6.
#   [FIX-6] prepare_simplices_cpu is type-stable (function barrier on the index
#           type; no P[collect(idx), :] / verts[i, :] temporaries).
#   [FIX-7] Balanced cyclic thread partition, and a serial path for small inputs
#           so that "many small polytopes" is not "many @threads task spawns".
#           Opt out explicitly with the new optional `threaded` keyword.

module CPUIntersection

using LinearAlgebra
using Combinatorics
using Base.Threads
using StaticArrays
using Random

export get_intersecting_pairs_cpu, simplices_intersect_sat_cpu, compute_face_normal, prepare_simplices_cpu
export check_intersections_range_cpu, find_intersections_in_subset

const MAXDIM = 10   # dimensions for which face tables are built

# ---------------------------------------------------------------------------
# Unrolled Laplace determinants (exact, division-free)
# ---------------------------------------------------------------------------

macro generate_determinants_up_to(n)
    n = Int(n)

    function make_det_func(k)
        fname = Symbol("det", k, "x", k)
        args = [Symbol(:m, i, j) for i in 1:k, j in 1:k]

        if k == 2
            return quote
                @inline function $(esc(fname))($(args...))
                    return m11 * m22 - m12 * m21
                end
            end
        end

        subcalls = Expr[]
        for j in 1:k
            subcols = filter(!=(j), 1:k)
            subargs = []
            for i in 2:k, c in subcols
                push!(subargs, Symbol(:m, i, c))
            end
            subcall = Expr(:call, Symbol("det", k - 1, "x", k - 1), subargs...)
            s = (-1)^(1 + j)
            push!(subcalls, :($(s) * $(Symbol(:m, 1, j)) * $subcall))
        end

        quote
            @inline function $(esc(fname))($(args...))
                $(foldl((x, y) -> :($x + $y), subcalls))
            end
        end
    end

    det_funcs = [make_det_func(k) for k in 2:n]
    quote
        $(det_funcs...)
    end
end

@generate_determinants_up_to 5

# ---------------------------------------------------------------------------
# Unrolled fraction-free (Bareiss) determinants
# ---------------------------------------------------------------------------

macro generate_determinants_bareiss_up_to(n)
    n = Int(n)

    function make_det_func(k)
        fname = Symbol("det", k, "x", k, "_bareiss")
        args  = [Symbol(:m, i, j) for i in 1:k, j in 1:k]
        stmts = Expr[]

        for i in 1:k, j in 1:k
            push!(stmts, :($(Symbol(:a_, i, "_", j)) = $(Symbol(:m, i, j))))
        end

        for pivot in 1:(k - 1)
            app = Symbol(:a_, pivot, "_", pivot)

            # [FIX-4] the guard used to be emitted only for `pivot < k - 1`,
            # leaving the last pivot unchecked -> division by zero.
            inner = :(return zero(Int64))
            for m in reverse((pivot + 1):k)
                swap_code = Expr[]
                for i in 1:k
                    ami = Symbol(:a_, m, "_", i)
                    api = Symbol(:a_, pivot, "_", i)
                    push!(swap_code, :(tmp = $ami; $ami = $api; $api = tmp))
                end
                amp = Symbol(:a_, m, "_", pivot)
                inner = quote
                    if $amp != 0
                        _sgn = -_sgn
                        $(swap_code...)
                    else
                        $inner
                    end
                end
            end
            push!(stmts, quote
                if $app == 0
                    $inner
                end
            end)

            for i in (pivot + 1):k
                aip = Symbol(:a_, i, "_", pivot)
                for j in (pivot + 1):k
                    aij   = Symbol(:a_, i, "_", j)
                    apj   = Symbol(:a_, pivot, "_", j)
                    denom = Symbol(:a_, pivot - 1, "_", pivot - 1)
                    upd   = :($aij * $app - $aip * $apj)
                    # [FIX-4] div, not Core.Intrinsics.sdiv_int: the intrinsic
                    # skips the zero and typemin/-1 checks and traps rather than
                    # throwing.  The guard above keeps the denominator nonzero.
                    push!(stmts, :($aij = $(pivot == 1 ? upd : :(div($upd, $denom)))))
                end
            end
        end

        push!(stmts, quote
            return _sgn * $(Symbol(:a_, k, "_", k))
        end)

        quote
            @inline function $(esc(fname))($(args...))
                _sgn = 1
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

# ---------------------------------------------------------------------------
# Generalized cross product
# ---------------------------------------------------------------------------
#
# Convention: the d-1 spanning vectors are the COLUMNS of a d x (d-1) matrix M;
# component i of the normal is (-1)^i * det(M with row i deleted).

macro generate_gcross_unrolled_full(d)
    d_val = Int(d)
    fname = Symbol("gcross", d_val)

    # vsym[i, j] = component i of vector j
    vsym       = [Symbol("v", i, "_", j) for i in 1:d_val, j in 1:(d_val - 1)]
    comp_exprs = Expr[]

    for i in 1:d_val
        rows_minor = filter(x -> x != i, 1:d_val)
        subargs = []
        for r in rows_minor, c in 1:(d_val - 1)
            push!(subargs, vsym[r, c])
        end
        # Division-free Laplace everywhere.  Bareiss squares its intermediates
        # (the update is a_ij*a_kk - a_ik*a_kj before the division), which for the
        # 5x5 case caps max|coordinate| at 53 instead of 274 in Int64.  Change the
        # threshold back to `>= 6` to use det5x5_bareiss, and shrink the entry in
        # _MAX_ABS_COORD accordingly.
        det_name = d_val >= 7 ? Symbol("det", d_val - 1, "x", d_val - 1, "_bareiss") :
                                Symbol("det", d_val - 1, "x", d_val - 1)
        call_expr = Expr(:call, det_name, subargs...)
        isodd(i) && (call_expr = :(-$call_expr))
        push!(comp_exprs, :($(Symbol("n", i)) = $call_expr))
    end

    # The parameter order is vec(vsym), i.e. COLUMN-MAJOR: all d components of
    # vector 1, then all d components of vector 2, ...  Call sites must match.
    scalar_args      = vec(vsym)
    scalar_func_name = Symbol(fname, "_scalar!")

    scalar_func = quote
        @inline function $(esc(scalar_func_name))($(scalar_args...))
            @inbounds begin
                $(comp_exprs...)
                return SVector{$d_val,Int64}($((Symbol("n", i) for i in 1:d_val)...))
            end
        end
    end

    vec_func = quote
        function $(esc(:_generalized_cross_product))(vs::SVector{$d_val - 1,SVector{$d_val,Int64}})
            $(Expr(:block, [:($(Symbol("v", i, "_", j)) = vs[$j][$i])
                            for i in 1:d_val, j in 1:(d_val - 1)]...))
            return $(esc(scalar_func_name))(
                $([Symbol("v", i, "_", j) for i in 1:d_val, j in 1:(d_val - 1)]...))
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

# Generic fallback for D > 6.  Slow (scratch matrix + LinearAlgebra.det_bareiss)
# but correct, and used as the reference in _selftest.
function _generalized_cross_product(vs::SVector{N,SVector{D,Int64}}) where {N,D}
    @assert N == D - 1
    # Int128 scratch: det_bareiss squares its intermediates, so Int64 here would
    # cap max|coordinate| at 16 for D = 7.  The narrowing conversion below throws
    # InexactError rather than wrapping if the result does not fit.
    tmp    = Matrix{Int128}(undef, N, N)
    normal = MVector{D,Int64}(undef)
    sgn    = 1
    @inbounds for i in 1:D
        r = 1
        for src in 1:D
            src == i && continue
            for j in 1:N
                tmp[r, j] = Int128(vs[j][src])
            end
            r += 1
        end
        sgn = -sgn
        normal[i] = Int64(sgn * LinearAlgebra.det_bareiss(tmp))
    end
    return SVector(normal)
end

# Kept so that any existing MVector call site still works.
@inline _generalized_cross_product(vs::MVector{N,SVector{D,Int64}}) where {N,D} =
    _generalized_cross_product(SVector(vs))

# Exposes the generalized cross product to the outside world for hyperplane
# computation.  `face_verts` holds D points; the D-1 spanning vectors are taken
# relative to the first.
function compute_face_normal(face_verts::AbstractVector{<:AbstractVector{<:Integer}}, ::Val{D}) where {D}
    p0 = SVector{D,Int64}(face_verts[1])
    span = SVector{D - 1,SVector{D,Int64}}(
        ntuple(t -> SVector{D,Int64}(face_verts[t + 1]) - p0, Val(D - 1)))
    return _generalized_cross_product(span)
end

# ---------------------------------------------------------------------------
# [FIX-2] Face tables.
#
# _FACEVERTS_{D}_{K}[i] is the sorted vertex-index tuple of the i-th K-face of a
# D-simplex.  These depend only on (D, K), never on the simplex, so they live
# here once instead of being rebuilt as a nested Vector forest per simplex.
# The K spanning vectors of a face F are verts[F[j+1]] - verts[F[1]].
# ---------------------------------------------------------------------------

const _FACEVERTS_RT = Dict{Tuple{Int,Int},Vector{Vector{Int}}}()

for D in 3:MAXDIM, K in 1:(D - 2)
    local tbl = [NTuple{K + 1,Int}(F) for F in combinations(1:(D + 1), K + 1)]
    @eval const $(Symbol("_FACEVERTS_", D, "_", K)) = $tbl
    _FACEVERTS_RT[(D, K)] = [collect(t) for t in tbl]
end

# ---------------------------------------------------------------------------
# Simplex  --  fully isbits, so Vector{Simplex} is one contiguous allocation
# ---------------------------------------------------------------------------

struct Simplex{V,D}
    verts::SVector{V,SVector{D,Int64}}          # V = D+1 vertices
    facet_normals::SVector{V,SVector{D,Int64}}  # outward; zero if degenerate
    lo::SVector{D,Int64}                        # AABB
    hi::SVector{D,Int64}
end

# [FIX-3] Every facet gets a slot.  The old code `continue`d past degenerate
# facets and then fed a short vector to SVector{D+1}, throwing DimensionMismatch.
function _facet_normals(verts::SVector{V,SVector{D,Int64}}) where {V,D}
    span = MVector{D - 1,SVector{D,Int64}}(undef)
    out  = MVector{V,SVector{D,Int64}}(undef)
    @inbounds for off in 1:V
        base = off == 1 ? 2 : 1
        c = 0
        for t in 1:V
            (t == off || t == base) && continue
            c += 1
            span[c] = verts[t] - verts[base]
        end
        n = _generalized_cross_product(SVector(span))
        if !iszero(n) && dot(n, verts[off] - verts[base]) > 0
            n = -n
        end
        out[off] = n
    end
    return SVector(out)
end

function compute_simplex_data(verts::SVector{V,SVector{D,Int64}}) where {V,D}
    lo = verts[1]
    hi = verts[1]
    @inbounds for t in 2:V
        lo = min.(lo, verts[t])
        hi = max.(hi, verts[t])
    end
    return Simplex(verts, _facet_normals(verts), lo, hi)
end

# ---------------------------------------------------------------------------
# Separation tests
# ---------------------------------------------------------------------------

@inline function axis_separates(s1_verts::SVector{V,SVector{D,Int64}},
                                s2_verts::SVector{V,SVector{D,Int64}},
                                axis) where {V,D}
    projs1 = ntuple(i -> dot(s1_verts[i], axis), Val(V))
    projs2 = ntuple(i -> dot(s2_verts[i], axis), Val(V))
    return maximum(projs1) <= minimum(projs2) || maximum(projs2) <= minimum(projs1)
end

# [FIX-5] Coordinate axes are legitimate candidate axes, so an AABB hit is a
# genuine separation certificate (and uses the same <= convention).
@inline _aabb_separates(s1::Simplex{V,D}, s2::Simplex{V,D}) where {V,D} =
    any(s1.hi .<= s2.lo) || any(s2.hi .<= s1.lo)

# [FIX-1] The scalar kernel wants its arguments grouped by VECTOR (all d
# components of spanning vector 1, then of vector 2, ...).  The old call site
# emitted a k x d comprehension, whose column-major splat groups by COMPONENT --
# correct for k = 1, scrambled for k >= 2.  The hoisted locals are now named
# v1_{component}_{vector} so the natural `for m in 1:d, j in 1:k` comprehension
# splats in the right order.
macro generate_cross_axes_case_scalar(d)
    d_val = Int(d)
    stmts = Expr[]
    scalar_func = Symbol("gcross", d_val, "_scalar!")

    for k in 1:(d_val - 2)
        l  = d_val - 1 - k
        nk = binomial(d_val + 1, k + 1)
        nl = binomial(d_val + 1, l + 1)
        tbl_k = Symbol("_FACEVERTS_", d_val, "_", k)
        tbl_l = Symbol("_FACEVERTS_", d_val, "_", l)

        push!(stmts, quote
            let idx_k = $tbl_k, idx_l = $tbl_l,
                w1 = $(esc(:s1_verts)), w2 = $(esc(:s2_verts))

                @inbounds for i in 1:$nk
                    fi = idx_k[i]
                    p1 = w1[fi[1]]
                    $([:($(Symbol("a_", j)) = w1[fi[$(j + 1)]] - p1) for j in 1:k]...)
                    $([:($(Symbol("v1_", m, "_", j)) = $(Symbol("a_", j))[$m])
                       for j in 1:k, m in 1:d_val]...)

                    for ii in 1:$nl
                        fj = idx_l[ii]
                        p2 = w2[fj[1]]
                        $([:($(Symbol("b_", j)) = w2[fj[$(j + 1)]] - p2) for j in 1:l]...)
                        $([:($(Symbol("v2_", m, "_", j)) = $(Symbol("b_", j))[$m])
                           for j in 1:l, m in 1:d_val]...)

                        axis = $scalar_func(
                            $([Symbol("v1_", m, "_", j) for m in 1:d_val, j in 1:k]...),
                            $([Symbol("v2_", m, "_", j) for m in 1:d_val, j in 1:l]...))

                        if !iszero(axis) && axis_separates(w1, w2, axis)
                            return false
                        end
                    end
                end
            end
        end)
    end

    return Expr(:block, stmts...)
end

# Generic cross-axis enumeration (D > 6, and the reference used by _selftest).
# Returns false iff a separating axis was found.
function _cross_axes_generic(w1::SVector{V,SVector{D,Int64}},
                             w2::SVector{V,SVector{D,Int64}}, ::Val{D}) where {V,D}
    span = MVector{D - 1,SVector{D,Int64}}(undef)
    for k in 1:(D - 2)
        l = D - 1 - k
        idx_k = _FACEVERTS_RT[(D, k)]
        idx_l = _FACEVERTS_RT[(D, l)]
        for fi in idx_k
            p1 = w1[fi[1]]
            @inbounds for t in 1:k
                span[t] = w1[fi[t + 1]] - p1
            end
            for fj in idx_l
                p2 = w2[fj[1]]
                @inbounds for t in 1:l
                    span[k + t] = w2[fj[t + 1]] - p2
                end
                axis = _generalized_cross_product(SVector(span))
                if !iszero(axis) && axis_separates(w1, w2, axis)
                    return false
                end
            end
        end
    end
    return true
end

function simplices_intersect_sat_cpu(s1::Simplex{V,D}, s2::Simplex{V,D}) where {V,D}
    _aabb_separates(s1, s2) && return false

    s1_verts = s1.verts
    s2_verts = s2.verts

    # facets
    @inbounds for i in 1:V
        n1 = s1.facet_normals[i]
        !iszero(n1) && axis_separates(s1_verts, s2_verts, n1) && return false
        n2 = s2.facet_normals[i]
        !iszero(n2) && axis_separates(s1_verts, s2_verts, n2) && return false
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
        return _cross_axes_generic(s1_verts, s2_verts, Val(D))
    end

    # We've enumerated and tested all possible axes but none of them
    # separate.  Therefore, the simplices must intersect.
    return true
end

# ---------------------------------------------------------------------------
# Building simplices
# ---------------------------------------------------------------------------

# Everything in this module is exact integer arithmetic, but it is Int64
# arithmetic, and Julia wraps silently on overflow.  For a D-simplex with
# coordinates bounded by B the widest intermediate is dot(normal, edge), i.e.
# D * 2B * |det of a (D-1)x(D-1) matrix with entries <= 2B|; bounding that
# determinant by min(Hadamard, Leibniz) gives the largest B that keeps every
# intermediate inside Int64:
const _MAX_ABS_COORD = Dict(3 => 577053, 4 => 12904, 5 => 1292, 6 => 274,
                            7 => 89, 8 => 38, 9 => 19, 10 => 11, 11 => 7, 12 => 5)

function _check_coord_bound(P::AbstractMatrix{<:Integer}, ::Val{D}) where {D}
    lim = get(_MAX_ABS_COORD, D, 0)
    b = 0
    @inbounds for x in P
        a = abs(Int64(x))
        a > b && (b = a)
    end
    b <= lim || error(
        "cpu_intersection: max|coordinate| = $b exceeds the Int64-safe bound $lim " *
        "for D = $D; intermediates would wrap silently.  Rescale/translate the " *
        "point set, or widen the vertex element type to Int128.")
    return nothing
end

# [FIX-6] Function barrier: specialises on typeof(idx), so an abstract
# eltype(S_indices) costs one dynamic call per simplex instead of poisoning
# every array access inside.  Also avoids the P[collect(idx), :] temporary.
@inline function _pack_verts(P::AbstractMatrix, idx, ::Val{V}, ::Val{D}) where {V,D}
    return SVector{V,SVector{D,Int64}}(
        ntuple(t -> SVector{D,Int64}(ntuple(c -> Int64(@inbounds P[idx[t], c]), Val(D))),
               Val(V)))
end

# Precompute the type conversion and the derived data for each simplex.
function prepare_simplices_cpu(P::Matrix{Int}, S_indices::Vector, ::Val{D}) where {D}
    _check_coord_bound(P, Val(D))
    num_simplices = length(S_indices)
    simplices = Vector{Simplex{D + 1,D}}(undef, num_simplices)
    for i in 1:num_simplices
        idx = S_indices[i]
        @assert length(idx) == D + 1 "expected each simplex to have d+1 vertices"
        @inbounds simplices[i] = compute_simplex_data(_pack_verts(P, idx, Val(D + 1), Val(D)))
    end
    return simplices
end

# ---------------------------------------------------------------------------
# Pairwise search
# ---------------------------------------------------------------------------

function _pairs_serial(simplices::Vector{Simplex{V,D}}) where {V,D}
    n = length(simplices)
    clauses = Vector{Vector{Int}}()
    @inbounds for i in 1:(n - 1)
        t1 = simplices[i]
        for j in (i + 1):n
            simplices_intersect_sat_cpu(t1, simplices[j]) && push!(clauses, [-i, -j])
        end
    end
    return clauses
end

# [FIX-7] Cyclic (not blocked) partition: with a blocked split thread 1 gets the
# rows with the most columns and everyone else idles.
function _pairs_threaded(simplices::Vector{Simplex{V,D}}) where {V,D}
    n  = length(simplices)
    nt = nthreads()
    bufs = [Vector{Vector{Int}}() for _ in 1:nt]
    @threads :static for t in 1:nt
        clauses = bufs[t]
        @inbounds for i in t:nt:(n - 1)
            t1 = simplices[i]
            for j in (i + 1):n
                simplices_intersect_sat_cpu(t1, simplices[j]) && push!(clauses, [-i, -j])
            end
        end
    end
    return reduce(vcat, bufs; init = Vector{Vector{Int}}())
end

"""
    get_intersecting_pairs_cpu_generic(P, S_indices, Val(D); threaded = nothing)

Returns `Vector{Vector{Int}}`, one `[-i, -j]` clause per intersecting pair --
unchanged from before.  `threaded` is optional: `nothing` (the default) runs
serially for small inputs and threaded otherwise.  Pass `false` when you are
already parallelising over polytopes at the call site, which avoids spawning
`nthreads()` tasks per polytope.
"""
function get_intersecting_pairs_cpu_generic(P::Matrix{Int}, S_indices::Vector, ::Val{D};
                                            threaded::Union{Nothing,Bool} = nothing) where {D}
    simplices::Vector{Simplex{D + 1,D}} = prepare_simplices_cpu(P, S_indices, Val(D))
    num_simplices = length(simplices)
    if num_simplices <= 1
        return Vector{Vector{Int}}()
    end
    use_threads = threaded === nothing ? (nthreads() > 1 && num_simplices >= 128) : threaded
    return use_threads ? _pairs_threaded(simplices) : _pairs_serial(simplices)
end

# --- helpers for incremental solving ----------------------------------------

# Checks intersections for a single simplex `idx1` against a range of other
# simplices.  Efficient for use in worker threads.
function check_intersections_range_cpu(
    simplices::Vector{Simplex{V,D}},
    idx1::Int,
    range_start::Int,
    range_end::Int
) where {V,D}
    conflicts = Vector{Vector{Int}}()
    range_start > range_end && return conflicts
    t1 = simplices[idx1]
    @inbounds for idx2 in range_start:range_end
        simplices_intersect_sat_cpu(t1, simplices[idx2]) && push!(conflicts, [-idx1, -idx2])
    end
    return conflicts
end

# Validates a specific subset of simplices (e.g. a candidate solution).
# Returns a list of conflicting pairs found within this subset.
function find_intersections_in_subset(
    simplices::Vector{Simplex{V,D}},
    indices::Vector{Int}
) where {V,D}
    conflicts = Vector{Vector{Int}}()
    n = length(indices)
    @inbounds for i in 1:n
        idx1 = indices[i]
        t1 = simplices[idx1]
        for j in (i + 1):n
            idx2 = indices[j]
            simplices_intersect_sat_cpu(t1, simplices[idx2]) && push!(conflicts, [-idx1, -idx2])
        end
    end
    return conflicts
end

# ---------------------------------------------------------------------------
# Differential self-test (not exported)
# ---------------------------------------------------------------------------

# Same predicate, routed exclusively through the generic (non-unrolled) path and
# without the AABB shortcut.  If the unrolled kernels disagree with this, the
# macro is wrong.
function _intersect_reference(s1::Simplex{V,D}, s2::Simplex{V,D}) where {V,D}
    @inbounds for i in 1:V
        n1 = s1.facet_normals[i]
        !iszero(n1) && axis_separates(s1.verts, s2.verts, n1) && return false
        n2 = s2.facet_normals[i]
        !iszero(n2) && axis_separates(s1.verts, s2.verts, n2) && return false
    end
    return _cross_axes_generic(s1.verts, s2.verts, Val(D))
end

"""
    CPUIntersection._selftest(D; n = 500, coords = -4:4)

Compares the unrolled kernels against the generic reference on random lattice
simplices.  Returns the number of mismatches; 0 is what you want.
"""
function _selftest(D::Int; n::Int = 500, coords = -4:4, rng = Random.default_rng())
    V = D + 1
    mk() = compute_simplex_data(
        SVector{V,SVector{D,Int64}}(ntuple(_ -> SVector{D,Int64}(rand(rng, coords, D)), Val(V))))
    bad = 0
    for _ in 1:n
        s1 = mk()
        s2 = mk()
        a = simplices_intersect_sat_cpu(s1, s2)
        b = _intersect_reference(s1, s2)
        if a != b
            bad += 1
            bad <= 3 && @warn "mismatch" D s1.verts s2.verts fast = a ref = b
        end
    end
    return bad
end

end # module

# ---------------------------------------------------------------------------
# NOTES
#
# Exactness: all arithmetic is integral, so there is no rounding anywhere -- but
# it is Int64, and Julia wraps silently.  prepare_simplices_cpu now refuses input
# whose coordinates exceed the per-dimension bound in _MAX_ABS_COORD (577053 at
# D = 3 down to 274 at D = 6 and 38 at D = 8).  To go beyond that, change the
# vertex element type from Int64 to Int128 throughout Simplex, the gcross
# kernels and axis_separates; the bounds then rise by a factor of about 2^(64/k)
# where k is the degree of the widest intermediate.
#
# Remaining per-call allocation is the clause list: one 2-element Vector{Int}
# per intersecting pair, which is dictated by the return type.  If you ever want
# it gone, _pairs_serial / _pairs_threaded can push (Int32, Int32) tuples into a
# caller-supplied buffer; that is a real API change, so it is not done here.
#
# Cost per pair at D = 6 is ~3300 candidate axes; the AABB prefilter is what
# makes that tolerable.  Next steps if it is still the bottleneck: deduplicate
# axes (many (k,l) face pairs span the same hyperplane), or add a grid /
# interval-tree broad phase over the AABBs.
# ---------------------------------------------------------------------------

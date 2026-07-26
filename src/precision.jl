"""
    Precision

Exact integer-overflow analysis for the GPU separating-axis kernels.

The kernels evaluate, for a pair of `d`-simplices with integer vertices:

  1. difference vectors     `v = p_j - p_0`                       (entries ≤ W)
  2. a normal vector        `n = gcross(v_1, …, v_{d-1})`,
     whose components are `(d-1)×(d-1)` determinants of such entries
  3. projections            `⟨x, n⟩` for each of the `2(d+1)` vertices

Step 3 produces the largest values, so the whole computation is safe in a
machine integer type `T` iff

      V(d) = Σ_c m_c · N_c  ≤  typemax(T)

where `m_c = max_i |x_{i,c}|`, and `N_c` bounds `|n_c|` via Hadamard's
inequality applied to the minor that omits column `c`:

      N_c = ⌈ ((Σ_j w_j²) − w_c²)^((d−1)/2) ⌉,     w_j = coordinate spread.

For an isotropic box (`w_j = W`) and coordinates normalised to `[0,W]^d`
this collapses to the closed form

      V(d) = d · ⌈(d−1)^((d−1)/2)⌉ · W^d = C(d) · W^d,
      C(3)=6,  C(4)=24,  C(5)=80,  C(6)=336.

Everything here is computed in `BigInt`, so the overflow check itself cannot
overflow.
"""
module Precision

# Plain `using Printf`, not `using ..Printf`: this module has to be includable
# from any parent, and `..Printf` would require the *parent* to have a `Printf`
# binding.  BasicComputations does not have one -- which is also why the old
# gpu_intersection_*d.jl modules' `using ..LinearAlgebra, ..Printf` could not
# have resolved when included from there.
using Printf

export GPU_MAX_DIM, PrecisionStatus,
       hadamard_factor, normal_bounds, value_bound,
       max_safe_width, max_safe_abs_coord,
       polytope_precision, check_gpu_precision,
       narrowest_safe_type, precision_table

"""
Highest ambient dimension the generated GPU kernels are enabled for.

This is a policy constant, not a limitation of the generator: the code emitter is
dimension-agnostic and its output has been checked against the exact reference up
to `D = 7`. Cost per pair grows steeply, though —

    D    axes/pair   emitted stmts   max W (Int32)   max W (Int64)
    3           44             122             710       1 154 107
    4          210             234              97          24 898
    5          862             432              30          24 898
    6        3 346             794              13             549
    7       12 756           1 486               7             179
    8       48 474           2 866               4              77

— and `Int64` in 7D will spill registers. Raise this if you need higher
dimensions; `GPUIntersection.MAXDIM` follows it, and anything above falls back to
the CPU backend with a warning.
"""
const GPU_MAX_DIM = 7

"""Integer types the kernels can be instantiated with, narrowest first."""
const CANDIDATE_TYPES = (Int32, Int64)

# --------------------------------------------------------------------------- #
#  exact integer helpers
# --------------------------------------------------------------------------- #

"""`⌈√n⌉`, exact, for `n ≥ 0`."""
function ceil_isqrt(n::Integer)
    n <= 0 && return zero(n)
    r = isqrt(n)
    return r * r == n ? r : r + one(r)
end

"""
    hadamard_factor(k) -> BigInt

`⌈k^(k/2)⌉ = ⌈√(k^k)⌉`, the Hadamard bound on a `k×k` determinant whose
entries are bounded by 1.  `H(1..5) = 1, 2, 6, 16, 56`.
"""
hadamard_factor(k::Integer) = k <= 0 ? big(1) : ceil_isqrt(big(k)^k)

# --------------------------------------------------------------------------- #
#  bounds
# --------------------------------------------------------------------------- #

"""
    normal_bounds(d, w) -> Vector{BigInt}

Per-component bound on the normal vector, given per-coordinate spreads `w`.
`|n_c| ≤ ⌈((Σ w_j²) − w_c²)^((d−1)/2)⌉` by Hadamard applied to the minor
omitting column `c`.
"""
function normal_bounds(d::Integer, w::AbstractVector{<:Integer})
    length(w) == d || throw(ArgumentError("length(w) = $(length(w)) ≠ d = $d"))
    W = big.(w)
    S = sum(x -> x^2, W; init = big(0))
    k = Int(d) - 1                       # Int exponent: BigInt^BigInt is not guaranteed
    return [ceil_isqrt((S - W[c]^2)^k) for c in 1:d]
end

"""
    value_bound(d, w, m) -> BigInt

Largest absolute value that can appear anywhere inside the kernel, for a point
set with per-coordinate spreads `w` and per-coordinate maxima `m = max|x|`.

Three quantities are covered:

* `Σ_c m_c·N_c` — vertex projections `⟨x, n⟩` (dominant term);
* `Σ_c w_c·N_c` — the `⟨n, p−p₀⟩` orientation test, if still present;
* `(d−1)·H(d−2)·W^(d−1)` — partial sums inside the cofactor expansion, which
  can slightly exceed the Hadamard bound on the finished determinant. This
  term is dominated by the first whenever `max m_c ≥ 1`, but is checked anyway.
"""
function value_bound(d::Integer, w::AbstractVector{<:Integer},
                                m::AbstractVector{<:Integer})
    d >= 2 || throw(ArgumentError("d must be ≥ 2, got $d"))
    N = normal_bounds(d, w)
    W = big.(w)
    M = big.(m)
    proj     = sum(c -> M[c] * N[c], 1:d; init = big(0))
    off_face = sum(c -> W[c] * N[c], 1:d; init = big(0))
    Wmax = isempty(W) ? big(0) : maximum(W)
    k = d - 1
    cofactor = k <= 1 ? Wmax : big(k) * hadamard_factor(k - 1) * Wmax^k
    return max(proj, off_face, cofactor)
end

"""Isotropic special case: spread `W` in every coordinate, `max|x| = M`."""
value_bound(d::Integer, W::Integer, M::Integer) =
    value_bound(d, fill(W, d), fill(M, d))

function _bisect(pred, hi::Integer)
    lo = big(0)
    hi = big(hi)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        pred(mid) ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

"""
    max_safe_width(d; T = Int64) -> BigInt

Largest bounding-box width `W` that is provably overflow-free in type `T`,
assuming coordinates have been translated into `[0,W]^d` (which the kernel
driver does; translation cannot change any comparison outcome).

    d │ Int32 │      Int64
    ──┼───────┼───────────
    3 │   710 │ 1 154 107
    4 │    97 │    24 898
    5 │    30 │     2 584
    6 │    13 │       549
"""
max_safe_width(d::Integer; T::Type = Int64) =
    _bisect(W -> value_bound(d, W, W) <= big(typemax(T)), big(10)^19)

"""
    max_safe_abs_coord(d; T = Int64) -> BigInt

Largest `max|x|` that is provably safe for *untranslated* coordinates, where
the spread may be as large as `2·max|x|`.  This is the relevant number for the
original `gpu_intersection_*d.jl` modules, which project raw coordinates.
"""
max_safe_abs_coord(d::Integer; T::Type = Int64) =
    _bisect(A -> value_bound(d, 2A, A) <= big(typemax(T)), big(10)^19)

# --------------------------------------------------------------------------- #
#  per-polytope status
# --------------------------------------------------------------------------- #

"""
Overflow status of a single polytope.

`bound` is the largest value the kernel can produce; `headroom` is
`typemax(Int64) / bound`.  `eltype` is the narrowest safe integer type, or
`nothing` if even `Int64` overflows.
"""
struct PrecisionStatus
    dim::Int
    width::Vector{Int}          # per-coordinate spread of the bounding box
    maxabs::Vector{Int}         # per-coordinate max |x| actually projected
    translated::Bool
    bound::BigInt
    eltype::Union{DataType, Nothing}
    headroom::Float64
end

fits(s::PrecisionStatus, ::Type{T}) where {T} = s.bound <= big(typemax(T))
Base.show(io::IO, s::PrecisionStatus) = print(io,
    "PrecisionStatus(dim=$(s.dim), W=$(maximum(s.width; init=0)), " *
    "bound≈$(@sprintf("%.3g", Float64(s.bound))), " *
    "eltype=$(s.eltype === nothing ? "OVERFLOW" : string(s.eltype)), " *
    "headroom≈$(@sprintf("%.3g", s.headroom)))")

"""
    polytope_precision(V; translate = true) -> PrecisionStatus

`V` is the vertex matrix (rows = vertices, columns = coordinates).  The lattice
points actually handed to the GPU lie in `conv(V)`, hence inside the bounding
box of `V`, so the bound computed here is valid for them too — no lattice point
enumeration is needed.

`translate = true` models the driver subtracting the per-coordinate minimum
(`max|x|` becomes the spread); `false` models the original modules, which
project raw coordinates.
"""
function polytope_precision(V::AbstractMatrix{<:Integer}; translate::Bool = true)
    d = size(V, 2)
    if size(V, 1) == 0 || d < 2
        # nothing a kernel would ever touch; report trivially safe
        return PrecisionStatus(d, zeros(Int, max(d, 0)), zeros(Int, max(d, 0)),
                               translate, big(0), Int32, Inf)
    end
    lo = [minimum(@view V[:, j]) for j in 1:d]
    hi = [maximum(@view V[:, j]) for j in 1:d]
    w  = Int[hi[j] - lo[j] for j in 1:d]
    m  = translate ? copy(w) :
                     Int[max(abs(lo[j]), abs(hi[j])) for j in 1:d]
    b  = value_bound(max(d, 2), w, m)
    ty = nothing
    for T in CANDIDATE_TYPES
        if b <= big(typemax(T))
            ty = T
            break
        end
    end
    hr = b == 0 ? Inf : Float64(big(typemax(Int64)) // b)
    return PrecisionStatus(d, w, m, translate, b, ty, hr)
end

"""Narrowest type that is safe for *every* polytope, or `nothing`."""
function narrowest_safe_type(stats::AbstractVector{PrecisionStatus})
    isempty(stats) && return Int32
    for T in CANDIDATE_TYPES
        all(s -> fits(s, T), stats) && return T
    end
    return nothing
end

# --------------------------------------------------------------------------- #
#  run-level check (this is what replaces the unconditional warning)
# --------------------------------------------------------------------------- #

"""
    check_gpu_precision(polytopes; translate = true, quiet = false)

Scan every polytope in the run and decide whether the GPU backend is exact.

Returns `(ok, eltype, stats)`:

* `ok`      — `true` if no polytope can overflow `Int64` and every dimension
              has a kernel;
* `eltype`  — narrowest integer type safe for the whole run (`Int32` is a
              worthwhile speedup on all current NVIDIA hardware, where 64-bit
              integer multiplies are emulated), or `nothing` if unsafe;
* `stats`   — per-polytope `PrecisionStatus`.

A warning is emitted **only** when something is actually unsafe, and it names
the offending polytope.
"""
function check_gpu_precision(polytopes::AbstractVector{<:AbstractMatrix{<:Integer}};
                             translate::Bool = true, quiet::Bool = false)
    isempty(polytopes) && return (true, Int32, PrecisionStatus[])
    stats = [polytope_precision(V; translate = translate) for V in polytopes]

    bad_dim = findall(s -> !(2 <= s.dim <= GPU_MAX_DIM), stats)
    bad_ovf = findall(s -> s.eltype === nothing, stats)
    ty      = narrowest_safe_type(stats)
    ok      = isempty(bad_dim) && isempty(bad_ovf)

    quiet && return (ok, ok ? ty : nothing, stats)

    if !isempty(bad_dim)
        dims = sort!(unique(stats[i].dim for i in bad_dim))
        @warn """
        GPU backend: $(length(bad_dim)) of $(length(stats)) polytopes have an \
        ambient dimension with no GPU kernel (dimensions $(dims); supported: \
        2–$(GPU_MAX_DIM)). These will need the CPU backend.
        """
    end

    if !isempty(bad_ovf)
        i = argmax([stats[j].bound for j in bad_ovf])
        worst = stats[bad_ovf[i]]
        idx = bad_ovf[i]
        lim = max_safe_width(worst.dim; T = Int64)
        @warn """
        GPU backend: integer overflow is POSSIBLE for $(length(bad_ovf)) of \
        $(length(stats)) polytopes — results for those are not trustworthy in \
        either direction.

          worst case: polytope #$idx, dimension $(worst.dim)
          bounding-box width      $(maximum(worst.width))  (per-axis: $(worst.width))
          kernel value bound      $(@sprintf("%.4g", Float64(worst.bound)))
          Int64 maximum           $(@sprintf("%.4g", Float64(typemax(Int64))))
          safe width in dim $(worst.dim)     $lim

        Either run these polytopes with intersection_backend="cpu", or rescale \
        / translate them into a smaller box first.
        """
    elseif ty !== nothing
        @info """
        GPU backend: overflow-free for all $(length(stats)) polytopes \
        (exact integer arithmetic in $ty; \
        min. headroom ×$(@sprintf("%.3g", minimum(s.headroom for s in stats)))).
        """
    end

    return (ok, ok ? ty : nothing, stats)
end

check_gpu_precision(V::AbstractMatrix{<:Integer}; kwargs...) =
    check_gpu_precision([V]; kwargs...)

"""
    assert_gpu_precision(P, d)

Cheap re-check immediately before a kernel launch, on the *actual* lattice
point matrix.  Needed because `check_full_dimensionality = true` may replace a
polytope by an HNF projection whose coordinates were never scanned up front.
"""
function assert_gpu_precision(P::AbstractMatrix{<:Integer}, d::Integer)
    s = polytope_precision(P)
    s.eltype === nothing && error("""
        GPU intersection kernel would overflow Int64 on this point set \
        (dimension $d, bounding-box width $(maximum(s.width)); safe width \
        $(max_safe_width(d))). This can happen after an HNF projection even \
        when the input vertices were small.""")
    return s.eltype
end

# --------------------------------------------------------------------------- #
#  documentation table
# --------------------------------------------------------------------------- #

"""Print the per-dimension safe-coordinate table."""
function precision_table(io::IO = stdout; maxdim::Integer = GPU_MAX_DIM)
    println(io, "Exact-arithmetic limits for the GPU separating-axis kernels")
    println(io, "V(d) = C(d)·W^d,  C(d) = d·⌈(d−1)^((d−1)/2)⌉,  W = bounding-box width\n")
    @printf(io, "%3s | %6s | %14s | %14s | %14s\n",
            "d", "C(d)", "max W (Int32)", "max W (Int64)", "max |x| (Int64,")
    @printf(io, "%3s | %6s | %14s | %14s | %14s\n",
            "", "", "translated", "translated", "untranslated)")
    println(io, "-"^72)
    for d in 3:maxdim
        C = big(d) * hadamard_factor(d - 1)
        @printf(io, "%3d | %6s | %14s | %14s | %14s\n",
                d, string(C), string(max_safe_width(d; T = Int32)),
                string(max_safe_width(d; T = Int64)),
                string(max_safe_abs_coord(d; T = Int64)))
    end
    return nothing
end

end # module

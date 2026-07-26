"""
    GPUIntersection

One module replacing `gpu_intersection_{3,4,5,6}d.jl`.

The dimension-specialised kernels are *generated*: `@generated` functions emit
fully unrolled, branch-free straight-line integer arithmetic for a given ambient
dimension `D`, element type `T` and tuning configuration. Nothing about `D` is
tested at run time — every loop bound, offset and coefficient is a compile-time
literal, exactly as in the old hand-written modules. `dump_pair_expr(D)` and
`dump_kernel_expr(D, T)` return the emitted code for inspection.

Generated per `(D, T, tuning)`:

* the generalised cross product as a subset-DP cofactor expansion
  (`Σ_s C(D,s)·s` multiplications instead of `D·(D-1)!` — 186 vs 720 at `D=6`);
* the **complete** separating-axis candidate set: all families `(p,q)` with
  `p + q = D-1`;
* overlap tests that skip the `p+q` provably-equal projections (25–36% fewer
  dot products);
* an axis-aligned bounding-box pre-filter;
* a tiled shared-memory pair loop with closed-form tile decoding.

REQUIREMENTS. `CUDA` and `Precision` must both be visible in the parent module
(`using ..CUDA`, `using ..Precision`), so include this file wherever the four
old modules were included — that is, in the same CUDA-conditional branch — and
`include("precision.jl"); using .Precision` *before* it.

See `GPU_NOTES.md` for derivations and for the differences from the originals.
"""
module GPUIntersection

using ..CUDA
using ..Precision

export get_intersecting_pairs_gpu,
       TuneConfig, tuning_path, load_tuning!, save_tuning, config_for,
       ensure_tuned!, quicktune!, quick_candidates, usable,
       axis_families, axis_count, shared_bytes,
       dump_pair_expr, dump_kernel_expr

# Single source of truth, shared with `Precision.check_gpu_precision` so the
# up-front warning and the actual dispatch can never disagree.
const MAXDIM = Precision.GPU_MAX_DIM

# =========================================================================== #
#  1.  Host-side combinatorics (evaluated only at code-generation time)
# =========================================================================== #

"""All `k`-subsets of `1:n`, lexicographic."""
function subsets(n::Int, k::Int)
    k < 0 && return Vector{Vector{Int}}()
    k == 0 && return [Int[]]
    k > n && return Vector{Vector{Int}}()
    out = Vector{Vector{Int}}()
    idx = collect(1:k)
    while true
        push!(out, copy(idx))
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in (i+1):k
            idx[j] = idx[j-1] + 1
        end
    end
    return out
end

"""
Number of *distinct* axis contributions from `k`-faces of one simplex. A 0-face
spans no directions, so all `D+1` vertices give the same axis and one
representative suffices.
"""
nfaces(D::Int, k::Int) = k == 0 ? 1 : binomial(D + 1, k + 1)

"""
    axis_families(D) -> Vector{Tuple{Int,Int}}

Candidate separating-axis families `(p,q)` with `p + q = D-1`: `p` direction
vectors from a `p`-face of `s1`, `q` from a `q`-face of `s2`; the axis is the
normal to their span. For polytopes with disjoint interiors some member of this
set is a separating axis, and the `p ≠ q` families are genuinely different, so
all `D` of them are needed.

Facet families `(D-1,0)` and `(0,D-1)` come first — cheapest, and they separate
most pairs, so the early exit fires sooner.

The originals were **incomplete for `D ≥ 5`**: the 5D kernel omitted `(3,1)`,
the 6D kernel omitted `(4,1)` and `(3,2)`.
"""
function axis_families(D::Int)
    mixed = [(p, D - 1 - p) for p in 1:(D-2)]
    sort!(mixed, by = f -> (nfaces(D, f[1]) * nfaces(D, f[2]), f[1]))
    return vcat([(D - 1, 0), (0, D - 1)], mixed)
end

"""Number of candidate axes in dimension `D`: 44, 210, 862, 3346."""
axis_count(D::Int) = sum(nfaces(D, p) * nfaces(D, q) for (p, q) in axis_families(D))

# --- per-simplex record layout -------------------------------------------- #
#   [1 .. (D+1)*D]        coordinates, column-major (D+1)×D
#   [.. +1 .. +D]         per-coordinate minimum  (AABB)
#   [.. +D+1 .. +2D]      per-coordinate maximum  (AABB)
reclen(D::Int) = (D + 1) * D + 2 * D
"""Shared-memory stride per simplex, forced odd so threads reading different
simplices at the same offset hit different banks."""
stride_of(D::Int) = reclen(D) | 1
cidx(D::Int, r::Int, j::Int) = (j - 1) * (D + 1) + r
minidx(D::Int, j::Int) = (D + 1) * D + j
maxidx(D::Int, j::Int) = (D + 1) * D + D + j

"""
    face_table(D) -> (tab, rowof)

Flat table, row stride `D+1`. Each row is a permutation of `1:(D+1)`: the `k+1`
vertices of a face followed by its complement. Storing the complement lets the
generator know statically how many projections are needed — every vertex of the
face projects to the same value, the axis being orthogonal to the face.
`rowof[k+1]` is the 0-based first row for `k`-faces.

Entries are vertex indices in `1:(D+1)`, so the table is stored in the *same*
element type as the coordinates. That lets it share one shared-memory
allocation, avoiding any question of whether two `CuStaticSharedArray` calls
with the same `(type, length)` alias.
"""
function face_table(D::Int)
    W = D + 1
    tab = Int[]
    rowof = zeros(Int, D + 1)
    row = 0
    for k in 0:(D-1)
        rowof[k+1] = row
        for f in (k == 0 ? [[1]] : subsets(W, k + 1))
            append!(tab, vcat(f, setdiff(1:W, f)))
            row += 1
        end
    end
    return tab, rowof
end

# =========================================================================== #
#  2.  Tuning
# =========================================================================== #

const _FACE_TABLE_CACHE = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()

"""
Memoised `face_table`, for run-time callers only.

`face_table` itself stays pure because `@generated` functions read it, and a
generated function must not depend on mutable global state. The run-time paths
(`shared_bytes`, `_run_kernels`) called it on every polytope, rebuilding the
table and its `subsets` allocations each time.
"""
face_table_cached(D::Int) = get!(() -> face_table(D), _FACE_TABLE_CACHE, D)

"""
Hardware-dependent kernel shape: `tile` simplices per tile side, `threads`
threads per block, inner axis loop unrolled `unroll`×, optional AABB
pre-filter, optional staging of the face table in shared memory.
"""
struct TuneConfig
    threads::Int
    tile::Int
    unroll::Int
    aabb::Bool
    shared_tables::Bool
end

TuneConfig(; threads = 256, tile = 32, unroll = 2, aabb = true,
             shared_tables = true) =
    TuneConfig(threads, tile, unroll, aabb, shared_tables)

"""Tuple form, so a config can be a `Val` type parameter (structs cannot)."""
cfgtuple(c::TuneConfig) = (c.threads, c.tile, c.unroll, c.aabb, c.shared_tables)

"""
The part of the config the *pair test* depends on. Keying `pair_intersects` on
only this avoids recompiling identical code for every `threads`/`tile`
combination the tuner tries.
"""
pairtuple(c::TuneConfig) = (c.unroll, c.aabb)

"""Static shared memory required per block, in bytes."""
function shared_bytes(D::Int, ::Type{T}, c::TuneConfig) where {T}
    n = 2 * c.tile * stride_of(D)
    c.shared_tables && (n += length(face_table_cached(D)[1]))
    return n * sizeof(T)
end

const DEFAULTS = Dict{Int, TuneConfig}(
    3 => TuneConfig(threads = 256, tile = 32, unroll = 4, aabb = true),
    4 => TuneConfig(threads = 256, tile = 32, unroll = 2, aabb = true),
    5 => TuneConfig(threads = 128, tile = 16, unroll = 2, aabb = true),
    6 => TuneConfig(threads = 128, tile = 16, unroll = 1, aabb = true),
    7 => TuneConfig(threads = 128, tile = 16, unroll = 1, aabb = true),
)

const TUNING = Dict{Tuple{Int, DataType}, TuneConfig}()
const _TUNING_LOADED = Ref(false)

tuning_dir() = get(ENV, "UNITRISAT_TUNING_DIR",
                   joinpath(get(ENV, "XDG_CACHE_HOME",
                                joinpath(homedir(), ".cache")), "UniTriSat"))

_slug(s) = replace(strip(String(s)), r"[^A-Za-z0-9]+" => "_")

"""Tuning-file path for the currently selected device."""
function tuning_path()
    dev = CUDA.device()
    cap = CUDA.capability(dev)
    return joinpath(tuning_dir(),
        "gpu_tuning_$(_slug(CUDA.name(dev)))_sm$(cap.major)$(cap.minor).conf")
end

_parsetype(s) = s == "Int32" ? Int32 : s == "Int64" ? Int64 :
    error("unknown element type \"$s\" in tuning file")

"""
    load_tuning!([path]) -> Int

Read a tuning file written by `tune_gpu.py` or `save_tuning`, whose lines are

    d=5 T=Int64 threads=128 tile=16 unroll=2 aabb=1 stab=1 gpairs=4.812

Unknown keys are ignored so the format can grow; later lines win, so an
appended entry supersedes an earlier one. Returns the number of entries read.
"""
function load_tuning!(path::AbstractString = tuning_path())
    isfile(path) || return 0
    n = 0
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        kv = Dict{String,String}()
        for tok in split(line)
            occursin('=', tok) || continue
            k, v = split(tok, '='; limit = 2)
            kv[k] = v
        end
        (haskey(kv, "d") && haskey(kv, "T")) || continue
        TUNING[(parse(Int, kv["d"]), _parsetype(kv["T"]))] = TuneConfig(
            threads       = parse(Int, get(kv, "threads", "256")),
            tile          = parse(Int, get(kv, "tile", "32")),
            unroll        = parse(Int, get(kv, "unroll", "2")),
            aabb          = get(kv, "aabb", "1") == "1",
            shared_tables = get(kv, "stab", "1") == "1")
        n += 1
    end
    empty!(_CONFIG_CACHE)
    return n
end

"""Append a tuned entry to the tuning file."""
function save_tuning(d::Int, ::Type{T}, c::TuneConfig, gpairs::Real;
                     path::AbstractString = tuning_path()) where {T}
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, "d=$d T=$T threads=$(c.threads) tile=$(c.tile) ",
                    "unroll=$(c.unroll) aabb=$(c.aabb ? 1 : 0) ",
                    "stab=$(c.shared_tables ? 1 : 0) gpairs=$gpairs")
    end
    TUNING[(d, T)] = c
    empty!(_CONFIG_CACHE)
    return path
end

function _lazy_load!()
    _TUNING_LOADED[] && return nothing
    try
        load_tuning!()
    catch err
        @debug "no GPU tuning file loaded" err
    end
    _TUNING_LOADED[] = true
    return nothing
end

_device_limit(attr, dflt) = try
    CUDA.attribute(CUDA.device(), attr)
catch
    dflt
end

"""Is this configuration usable on the current device?"""
function usable(D::Int, ::Type{T}, c::TuneConfig) where {T}
    c.threads >= 32 || return false
    c.tile * c.tile >= c.threads || return false      # else threads sit idle
    lim = _device_limit(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 48 * 1024)
    maxth = _device_limit(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1024)
    return shared_bytes(D, T, c) <= lim && c.threads <= maxth
end

const _CONFIG_CACHE = Dict{Tuple{Int, DataType}, TuneConfig}()

"""
Configuration for `(D, T)`, shrunk until it fits the device.

Memoised: the uncached path makes two `CUDA.attribute` driver calls plus a
`shared_bytes` table build per invocation, and it was invoked once per polytope.
`load_tuning!`, `save_tuning` and `quicktune!` invalidate the cache.
"""
config_for(D::Int, ::Type{T}) where {T} =
    get!(() -> _config_for_uncached(D, T), _CONFIG_CACHE, (D, T))

function _config_for_uncached(D::Int, ::Type{T}) where {T}
    _lazy_load!()
    c = get(TUNING, (D, T), get(DEFAULTS, D, TuneConfig()))
    while !usable(D, T, c) && c.tile > 4
        c = TuneConfig(min(c.threads, c.tile * c.tile ÷ 4), c.tile ÷ 2,
                       c.unroll, c.aabb, c.shared_tables)
    end
    if !usable(D, T, c) && c.shared_tables
        c = TuneConfig(c.threads, c.tile, c.unroll, c.aabb, false)
    end
    if !usable(D, T, c)
        c = TuneConfig(64, 8, 1, c.aabb, false)
    end
    return c
end

"""
Candidates for the quick in-process search: the built-in default for this
dimension plus its neighbours along each knob. Small on purpose — every
candidate costs one CUDA kernel compilation.
"""
function quick_candidates(D::Int)
    b = get(DEFAULTS, D, TuneConfig())
    out = TuneConfig[b]
    for th in (128, 256, 512)
        th == b.threads || push!(out, TuneConfig(th, b.tile, b.unroll, b.aabb, b.shared_tables))
    end
    for ti in (16, 32)
        ti == b.tile || push!(out, TuneConfig(b.threads, ti, b.unroll, b.aabb, b.shared_tables))
    end
    # `unroll` is the only knob `pair_intersects` depends on, and `_kernel`
    # inlines it, so every extra unroll value recompiles the whole generated
    # body.  In 6D that body is thousands of statements, so leave it alone there
    # and let `tune_gpu.py` explore it offline if desired.
    if D <= 5
        for u in (D <= 4 ? (1, 2, 4) : (1, 2))
            u == b.unroll || push!(out, TuneConfig(b.threads, b.tile, u, b.aabb, b.shared_tables))
        end
    end
    return out
end

# =========================================================================== #
#  3.  Code generation
# =========================================================================== #

_sym(parts...) = Symbol(join(parts, "_"))

"""Fold expressions into a left-associated sum."""
function sumexpr(terms::Vector)
    isempty(terms) && return 0
    e = terms[1]
    for t in terms[2:end]
        e = :($e + $t)
    end
    return e
end

"""
Emit the generalised cross product of the `D-1` vectors named `<v>_<i>_<j>`
into `<out>_<j>`, `j = 1..D`.

Subset DP: `M[S]` is the determinant of rows `1..|S|` and columns `S`, built by
expanding along the last row, so sub-minors are shared:

    M[S] = Σ_t (-1)^(|S|+t) · v_{|S|}[j_t] · M[S \\ {j_t}],
    n_c  = (-1)^(c+1) · M[{1..D} \\ {c}].

Verified against direct minor determinants, and for `⟨n, vᵢ⟩ = 0`, in
dimensions 2–7.
"""
function emit_gcross!(stmts::Vector, D::Int, v::Symbol, out::Symbol)
    prev = Dict{Vector{Int}, Any}(Int[] => 1)
    for s in 1:(D-1)
        cur = Dict{Vector{Int}, Any}()
        for S in subsets(D, s)
            terms = Any[]
            for (t, j) in enumerate(S)
                coef = prev[filter(!=(j), S)]
                vv = _sym(v, s, j)
                term = coef === 1 ? vv : :($vv * $coef)
                push!(terms, iseven(s + t) ? term : :(-$term))
            end
            sym = _sym(:m, s, S...)
            push!(stmts, :($sym = $(sumexpr(terms))))
            cur[S] = sym
        end
        prev = cur
    end
    for c in 1:D
        src = prev[filter(!=(c), collect(1:D))]
        push!(stmts, :($(_sym(out, c)) = $(isodd(c) ? src : :(-$src))))
    end
    return stmts
end

"""
`@inbounds`-annotated read of `arr[base + off]`.

Annotating each access rather than wrapping the whole test in one `@inbounds`
region: the test contains `return false` statements, and a region with early
exits is a construct worth not relying on.
"""
shraw(arr::Symbol, base, off) = :(@inbounds $arr[$base + $off])

"""Read coordinate `j` (a literal) of run-time vertex index `r`."""
shc(arr::Symbol, base, D::Int, r, j::Int) =
    shraw(arr, base, :($(Int32((j - 1) * (D + 1))) + $r))

"""
Emit the overlap test for one axis. Only `a₀` plus the `D-p` complementary
vertices of each face are projected.

The sign of the axis is irrelevant — `hi1 ≤ lo2 || hi2 ≤ lo1` is invariant
under `n → -n` — so no orientation fix-up is emitted; the originals computed
one and discarded it.
"""
function emit_axis_test!(stmts::Vector, D::Int, n::Symbol, sh::Symbol,
                         o1::Symbol, o2::Symbol,
                         a::Vector{Symbol}, b::Vector{Symbol}, p::Int, q::Int)
    dotp(base, r) = sumexpr(Any[:($(shc(sh, base, D, r, j)) * $(_sym(n, j)))
                                for j in 1:D])
    reps1 = vcat([a[1]], a[(p+2):(D+1)])
    reps2 = vcat([b[1]], b[(q+2):(D+1)])
    body = Any[]
    for (t, r) in enumerate(reps1)
        push!(body, :($(_sym(:pa, t)) = $(dotp(o1, r))))
    end
    for (t, r) in enumerate(reps2)
        push!(body, :($(_sym(:pb, t)) = $(dotp(o2, r))))
    end
    red(f, pre, m) = foldl((x, y) -> :($f($x, $y)), Any[_sym(pre, t) for t in 1:m])
    append!(body, Any[
        :($(_sym(:lo, 1)) = $(red(:min, :pa, length(reps1)))),
        :($(_sym(:hi, 1)) = $(red(:max, :pa, length(reps1)))),
        :($(_sym(:lo, 2)) = $(red(:min, :pb, length(reps2)))),
        :($(_sym(:hi, 2)) = $(red(:max, :pb, length(reps2)))),
        :(if $(_sym(:hi, 1)) <= $(_sym(:lo, 2)) ||
             $(_sym(:hi, 2)) <= $(_sym(:lo, 1))
              return false
          end)])
    # bitwise | rather than ||: no short-circuit branches in the hot path
    nz = foldl((x, y) -> :($x | $y), Any[:($(_sym(n, j)) != 0) for j in 1:D])
    push!(stmts, :(if $nz
                       $(Expr(:block, body...))
                   end))
    return stmts
end

"""
Emit the body of `pair_intersects`: AABB pre-filter, then every axis family.

`false` means a separating axis exists (interiors disjoint); falling through to
`true` means the interiors overlap.

`ft`/`ftoff` are the array and base offset holding the face table — either the
shared array itself (offset past the coordinate tiles) or the global table
(offset 0). Because the function is inlined and the caller passes a literal
offset, the addressing folds to constants either way.
"""
function emit_pair_test(D::Int, unroll::Int, aabb::Bool)
    sh, ft, o1, o2, ftoff = :sh, :ft, :o1, :o2, :ftoff
    W = D + 1
    _, rowof = face_table(D)
    stmts = Any[Expr(:meta, :inline)]

    if aabb
        for j in 1:D
            push!(stmts, :(if $(shraw(sh, o1, Int32(maxidx(D, j)))) <=
                              $(shraw(sh, o2, Int32(minidx(D, j)))) ||
                              $(shraw(sh, o2, Int32(maxidx(D, j)))) <=
                              $(shraw(sh, o1, Int32(minidx(D, j))))
                               return false
                           end))
        end
    end

    for (p, q) in axis_families(D)
        NA, NB = nfaces(D, p), nfaces(D, q)
        a = [_sym(:a, i) for i in 1:W]
        b = [_sym(:b, i) for i in 1:W]

        # hoisted out of the inner loop
        loadA = Any[:($(a[i]) = $(shraw(ft, ftoff,
            :($(Int32(rowof[p+1] * W + i)) + (ia - 1) * $(Int32(W)))))) for i in 1:W]
        dirA = Any[]
        for i in 1:p, j in 1:D
            push!(dirA, :($(_sym(:u, i, j)) =
                $(shc(sh, o1, D, a[i+1], j)) - $(shc(sh, o1, D, a[1], j))))
        end

        inner = Any[:($(b[i]) = $(shraw(ft, ftoff,
            :($(Int32(rowof[q+1] * W + i)) + (ib - 1) * $(Int32(W)))))) for i in 1:W]
        for i in 1:q, j in 1:D
            push!(inner, :($(_sym(:u, p + i, j)) =
                $(shc(sh, o2, D, b[i+1], j)) - $(shc(sh, o2, D, b[1], j))))
        end
        emit_gcross!(inner, D, :u, :n)
        emit_axis_test!(inner, D, :n, sh, o1, o2, a, b, p, q)

        # Emit `inner` as few times as possible.  At unroll = 1 a plain loop
        # suffices; the scalar remainder is emitted only when the trip count is
        # not a multiple of the unroll factor.  Each unrolled copy lives in its
        # own `let`, so the temporaries cannot collide.
        U = max(1, min(unroll, NB))
        main = NB - NB % U
        loops = Any[]
        if U == 1
            push!(loops, :(for ib in 1:$NB
                               $(Expr(:block, inner...))
                           end))
        else
            unrolled = Any[:(let ib = ib0 + $t
                                 $(Expr(:block, inner...))
                             end) for t in 0:(U-1)]
            push!(loops, :(for ib0 in 1:$U:$main
                               $(Expr(:block, unrolled...))
                           end))
            main < NB && push!(loops, :(for ib in $(main + 1):$NB
                                            $(Expr(:block, inner...))
                                        end))
        end
        push!(stmts, :(for ia in 1:$NA
                           $(Expr(:block, loadA...))
                           $(Expr(:block, dirA...))
                           $(Expr(:block, loops...))
                       end))
    end

    push!(stmts, :(return true))
    return Expr(:block, stmts...)
end

@generated function pair_intersects(sh, ft, ftoff::Int32, o1::Int32, o2::Int32,
                                   ::Val{D}, ::Val{PC}) where {D, PC}
    emit_pair_test(D, PC[1], PC[2])
end

# =========================================================================== #
#  4.  Kernel
# =========================================================================== #

"""
Closed-form decode of a 1-based strict-upper-triangle pair index over `m` items
into `(i, j)`, `i < j`. Replaces the originals' `O(n)` search loop, which ran
about `0.29·n` iterations *per thread*.

`Float64` `sqrt` plus at most one correction step — verified exhaustively for
`m ≤ 199` and by spot checks to `m = 3·10⁶`.
"""
@inline function decode_pair(idx::Int64, m::Int64)
    b = 2.0 * Float64(m) - 1.0
    disc = b * b - 8.0 * Float64(idx)
    disc < 0.0 && (disc = 0.0)
    i = unsafe_trunc(Int64, (b - sqrt(disc)) * 0.5) + Int64(1)
    i = min(max(i, Int64(1)), m - Int64(1))
    while i > Int64(1) && (((i - Int64(1)) * (Int64(2) * m - i)) >> 1) >= idx
        i -= Int64(1)
    end
    while i < m - Int64(1) && ((i * (Int64(2) * m - i - Int64(1))) >> 1) < idx
        i += Int64(1)
    end
    return i, i + idx - (((i - Int64(1)) * (Int64(2) * m - i)) >> 1)
end

function emit_kernel(D::Int, ::Type{T}, c::TuneConfig) where {T}
    TS, ST, RL = c.tile, stride_of(D), reclen(D)
    TAB = length(face_table(D)[1])
    COORDS = 2 * TS * ST
    PC = pairtuple(c)
    body = Any[]

    # ONE shared allocation.  Two CuStaticSharedArray calls could in principle
    # be given the same underlying shared global when type and length coincide,
    # so the face table is stored in element type T and appended here instead.
    if c.shared_tables
        push!(body, :(sh = CUDA.CuStaticSharedArray($T, $(COORDS + TAB))))
        push!(body, quote
            let k = Int64(threadIdx().x)
                while k <= $(Int64(TAB))
                    @inbounds sh[$(Int64(COORDS)) + k] = ftab[k]
                    k += Int64(blockDim().x)
                end
            end
        end)
        ftexpr, ftoffexpr = :sh, Int32(COORDS)
    else
        push!(body, :(sh = CUDA.CuStaticSharedArray($T, $COORDS)))
        ftexpr, ftoffexpr = :ftab, Int32(0)
    end

    push!(body, quote
        nth = Int64(blockDim().x)
        tid = Int64(threadIdx().x)
        ti, tj2 = decode_pair(tp_base + Int64(blockIdx().x), ntiles + Int64(1))
        base_i = (ti - Int64(1)) * $(Int64(TS))
        base_j = (tj2 - Int64(2)) * $(Int64(TS))

        # cooperative, coalesced staging of both tiles
        let k = tid
            while k <= $(Int64(TS * RL))
                s = (k - Int64(1)) ÷ $(Int64(RL))
                e = (k - Int64(1)) % $(Int64(RL)) + Int64(1)
                gi = base_i + s + Int64(1)
                gj = base_j + s + Int64(1)
                @inbounds sh[s * $(Int64(ST)) + e] =
                    gi <= n ? sim[e, gi] : zero($T)
                @inbounds sh[($(Int64(TS)) + s) * $(Int64(ST)) + e] =
                    gj <= n ? sim[e, gj] : zero($T)
                k += nth
            end
        end
        # every thread reaches this: no early return above it
        sync_threads()

        let lp = tid
            while lp <= $(Int64(TS * TS))
                li = (lp - Int64(1)) ÷ $(Int64(TS))
                lj = (lp - Int64(1)) % $(Int64(TS))
                i1 = base_i + li + Int64(1)
                i2 = base_j + lj + Int64(1)
                if i1 < i2 && i2 <= n
                    if pair_intersects(sh, $ftexpr, $ftoffexpr,
                                       Int32(li * $(Int32(ST))),
                                       Int32(($(Int32(TS)) + lj) * $(Int32(ST))),
                                       Val($D), Val($PC))
                        # Returns the OLD value, so `slot` is a 0-based index.
                        # CUDA.jl's own docstring calls `@atomic` experimental
                        # ("might change without warning") and points at the
                        # atomic_...! functions for a stable API; `@atomic a[i] +=
                        # v` lowers to exactly this call, and Int64 is a natively
                        # supported type.
                        slot = CUDA.atomic_add!(pointer(counter, 1), Int64(1))
                        if slot < cap
                            @inbounds out[slot + Int64(1), 1] = Int32(i1)
                            @inbounds out[slot + Int64(1), 2] = Int32(i2)
                        end
                    end
                end
                lp += nth
            end
        end
        return nothing
    end)
    return Expr(:block, body...)
end

@generated function _kernel(sim::CUDA.CuDeviceArray{T}, ftab, n::Int64,
                            out, counter, cap::Int64,
                            tp_base::Int64, ntiles::Int64,
                            ::Val{D}, ::Val{CT}) where {T, D, CT}
    emit_kernel(D, T, TuneConfig(CT...))
end

"""Generated pair-test source, for inspection: `dump_pair_expr(5)`."""
dump_pair_expr(D::Int; c::TuneConfig = get(DEFAULTS, D, TuneConfig())) =
    emit_pair_test(D, c.unroll, c.aabb)

"""Generated kernel source, for inspection: `dump_kernel_expr(5, Int64)`."""
dump_kernel_expr(D::Int, ::Type{T} = Int64;
                 c::TuneConfig = get(DEFAULTS, D, TuneConfig())) where {T} =
    emit_kernel(D, T, c)

# =========================================================================== #
#  5.  Host driver
# =========================================================================== #

"""
Build the flat record matrix, translating the point set into `[0, w]^d`.

Translating by a common vector shifts every projection by the same `⟨t, n⟩`, so
no comparison can change — but it replaces `max|x|` by the bounding-box width in
the overflow bound, which is what makes the `Int32` fast path reachable for
polytopes far from the origin.
"""
function build_records(P::AbstractMatrix{<:Integer},
                       S::AbstractVector{<:NTuple{K, Int}},
                       ::Type{T}) where {K, T}
    D = size(P, 2)
    K == D + 1 ||
        throw(ArgumentError("simplices have $K vertices, expected $(D+1) in dim $D"))
    lo = [minimum(@view P[:, j]) for j in 1:D]
    R = Matrix{T}(undef, reclen(D), length(S))
    @inbounds for (s, tup) in enumerate(S)
        for j in 1:D
            mn, mx = typemax(T), typemin(T)
            for r in 1:K
                v = T(P[tup[r], j] - lo[j])
                R[cidx(D, r, j), s] = v
                v < mn && (mn = v)
                v > mx && (mx = v)
            end
            R[minidx(D, j), s] = mn
            R[maxidx(D, j), s] = mx
        end
    end
    return R
end

"""
    _run_kernels(R, D, c, n, capacity, collect) -> (pairs, count)

Batched launch loop, shared by the public entry point and the in-process tuner.
With `collect = false` only the counter is read back, which is what makes it
usable as a benchmark. Returns pairs as `(i, j)` tuples, sorted per batch, so
the result is deterministic for a given tiling despite the atomic append.
"""
function _run_kernels(R::Matrix{T}, D::Int, c::TuneConfig, n::Int,
                      capacity::Int, collect::Bool) where {T}
    ntiles = cld(n, c.tile)
    ntilepairs = ntiles * (ntiles + 1) ÷ 2
    # a block emits at most tile² pairs, so this batch size rules out overflow
    per_batch = max(1, capacity ÷ (c.tile * c.tile))
    bufrows = min(capacity, ntilepairs * c.tile * c.tile)

    sim = CuArray(R)
    ftab = CuArray(T.(face_table_cached(D)[1]))
    out = CuArray{Int32}(undef, bufrows, 2)   # only rows 1:got are ever read
    counter = CUDA.zeros(Int64, 1)
    pairs = Tuple{Int, Int}[]
    total = 0

    try
        done = 0
        while done < ntilepairs
            nb = min(per_batch, ntilepairs - done)
            fill!(counter, Int64(0))
            @cuda threads=c.threads blocks=nb _kernel(
                sim, ftab, Int64(n), out, counter, Int64(bufrows),
                Int64(done), Int64(ntiles), Val(D), Val(cfgtuple(c)))
            got = Int(Array(counter)[1])   # plain D2H copy; no scalar-indexing path
            got > bufrows &&
                error("internal error: output overflow ($got > $bufrows)")
            total += got
            if collect && got > 0
                host = Array(view(out, 1:got, :))
                batch = [(Int(host[r, 1]), Int(host[r, 2])) for r in 1:got]
                sort!(batch)
                append!(pairs, batch)
            end
            done += nb
        end
    finally
        # unsafe_free! returns these buffers to CUDA.jl's memory pool, where the
        # next call reuses them.  Do NOT call CUDA.reclaim() here: that hands the
        # pool back to the driver, so every following allocation becomes a fresh
        # cuMemAlloc.  With one call per polytope that dominated everything --
        # it made small 3D runs about two orders of magnitude slower than the
        # original modules, which never reclaimed.
        CUDA.unsafe_free!(sim); CUDA.unsafe_free!(ftab)
        CUDA.unsafe_free!(out);  CUDA.unsafe_free!(counter)
    end
    return pairs, total
end

"""
    quicktune!(D, T, R, n, capacity) -> TuneConfig

Benchmark `quick_candidates(D)` **on the caller's own data** — a subsample of the
record matrix about to be processed — and cache the winner. Real simplices beat
synthetic ones here, because how often the early exit fires depends on the
geometry.

In-process: no subprocess, no second CUDA context, no Python at run time.
`tune_gpu.py` remains the tool for a thorough offline search.
"""
function quicktune!(D::Int, ::Type{T}, R::Matrix{T}, n::Int,
                    capacity::Int) where {T}
    nb = min(n, 1200)
    Rs = nb == n ? R : R[:, 1:nb]
    best = nothing
    best_t = typemax(UInt64)
    for cand in quick_candidates(D)
        usable(D, T, cand) || continue
        try
            _run_kernels(Rs, D, cand, nb, capacity, false)   # warm-up = compile
            CUDA.synchronize()
            t0 = time_ns()
            _run_kernels(Rs, D, cand, nb, capacity, false)
            CUDA.synchronize()
            dt = time_ns() - t0
            if dt < best_t
                best, best_t = cand, dt
            end
        catch err
            @debug "tuning candidate failed" cand err
        end
    end
    best === nothing && return config_for(D, T)
    gpairs = (nb * (nb - 1) / 2) / (best_t / 1e9) / 1e9
    TUNING[(D, T)] = best
    empty!(_CONFIG_CACHE)
    try
        path = save_tuning(D, T, best, round(gpairs; digits = 6))
        @debug "GPU tuning for dim $D / $T: $best " *
               "($(round(gpairs; digits = 2)) Gpair/s), saved to $path"
    catch err
        # Loud, not @debug: if the cache cannot be written, every future session
        # re-tunes, which is exactly the symptom that is hard to attribute.
        @warn """
        GPU tuning succeeded but could not be saved, so it will be repeated in
        every future session. Set UNITRISAT_TUNING_DIR to a writable directory,
        or UNITRISAT_AUTOTUNE=0 to use the built-in defaults.
        """ path=tuning_path() exception=err
    end
    return best
end

const _TUNE_ATTEMPTED = Set{Tuple{Int, DataType}}()
const _TUNE_LOCK = ReentrantLock()

"""
    ensure_tuned!(D, T, R, n, capacity) -> Bool

"Tune on first run": the first time a `(dimension, element type)` pair is seen on
this device, run `quicktune!` and cache the result under
`\$XDG_CACHE_HOME/UniTriSat/`. Later calls, and later Julia sessions, read the
cache and do nothing.

Each `(D, T)` is attempted **at most once per session even on failure**, so a run
over thousands of polytopes cannot repeat the work. Polytopes with fewer than 200
simplices leave the attempt unspent, so tuning is not calibrated on unmeasurable
data. `UNITRISAT_AUTOTUNE=0` disables it; failure falls back to `DEFAULTS`.
"""
function ensure_tuned!(D::Int, ::Type{T}, R::Matrix{T}, n::Int,
                       capacity::Int) where {T}
    _lazy_load!()
    haskey(TUNING, (D, T)) && return false
    get(ENV, "UNITRISAT_AUTOTUNE", "1") == "1" || return false
    # Only calibrate on a workload big enough to be worth measuring *and* to
    # resemble the runs where throughput matters. Below this the timings are
    # dominated by launch overhead and would tune for the wrong regime; leaving
    # the attempt unspent lets a later, larger polytope trigger it.
    n < 2000 && return false
    return lock(_TUNE_LOCK) do
        (haskey(TUNING, (D, T)) || (D, T) in _TUNE_ATTEMPTED) && return false
        push!(_TUNE_ATTEMPTED, (D, T))
        # Exactly one message per tuning event, emitted before the work so the
        # pause is explained; the chosen configuration lands in the cache file.
        @info """
        GPU autotuning, dimension $D / $T, on $(CUDA.name(CUDA.device())) — \
        one-off, a few seconds in low dimensions. Result is cached in \
        $(tuning_path()); UNITRISAT_AUTOTUNE=0 disables tuning.
        """
        try
            quicktune!(D, T, R, n, capacity)
        catch err
            @warn "GPU autotuning failed; using built-in defaults" err
            return false
        end
        return haskey(TUNING, (D, T))
    end
end

"""
    get_intersecting_pairs_gpu(P, S_indices; eltype = nothing, capacity = 1<<22)

Return the two-literal exclusion clauses `[-i, -j]` for every pair of simplices
whose **interiors** intersect. `P` is the lattice-point matrix (rows = points),
`S_indices` a vector of `(d+1)`-tuples of row indices into `P`.

`eltype` defaults to the narrowest integer type `Precision` can prove exact for
this point set; pass `Int64` to force it.
"""
function get_intersecting_pairs_gpu(P::AbstractMatrix{<:Integer},
                                    S::AbstractVector{<:NTuple{K, Int}};
                                    eltype::Union{Type, Nothing} = nothing,
                                    capacity::Int = 1 << 22) where {K}
    n = length(S)
    n < 2 && return Vector{Vector{Int}}()
    D = size(P, 2)
    2 <= D <= MAXDIM ||
        error("GPUIntersection supports dimensions 2–$MAXDIM, got $D")

    T = eltype === nothing ? Precision.assert_gpu_precision(P, D) : eltype
    R = build_records(P, S, T)
    ensure_tuned!(D, T, R, n, capacity)
    pairs, _ = _run_kernels(R, D, config_for(D, T), n, capacity, true)
    return [[-i, -j] for (i, j) in pairs]
end

end # module

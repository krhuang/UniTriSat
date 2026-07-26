#!/usr/bin/env julia
#
# tune_worker.jl — long-lived benchmark server for tune_gpu.py.
#
# Speaks one request per stdin line, one reply per stdout line, so the Python
# driver pays Julia's and CUDA's start-up cost exactly once instead of once per
# candidate configuration.
#
#   bench   d=5 T=Int64 threads=128 tile=16 unroll=2 aabb=1 stab=1 n=3000 w=3 seed=1
#   verify  d=5 n=60 w=3 seed=1
#   quit
#
# Replies:
#   ready   name=... sm=86 shared=49152 sms=48 mem=...
#   ok      gpairs=4.812 ms=12.34 pairs=4498500 hits=120345
#   fail    <reason>
#
# Run standalone with:  julia --project=. tune_worker.jl
#
# stdout carries the protocol only; diagnostics go to stderr.

module TuneHarness

using CUDA, Printf, LinearAlgebra, Random

const HERE = @__DIR__

"""
Locate a source file without assuming a layout: `precision.jl` normally lives in
`src/` while `gpu_intersection.jl` lives in `src/Intersection_backends/`, and this
script may sit in either. Set `UNITRISAT_SRC` to override.
"""
function locate(name::AbstractString)
    cands = String[]
    if haskey(ENV, "UNITRISAT_SRC")
        push!(cands, ENV["UNITRISAT_SRC"],
              joinpath(ENV["UNITRISAT_SRC"], "Intersection_backends"))
    end
    append!(cands, [HERE, joinpath(HERE, "Intersection_backends"),
                    dirname(HERE), joinpath(dirname(HERE), "Intersection_backends"),
                    joinpath(dirname(HERE), "src"),
                    joinpath(dirname(HERE), "src", "Intersection_backends")])
    for d in cands
        f = joinpath(d, name)
        isfile(f) && return f
    end
    error("could not find $name near $HERE - set UNITRISAT_SRC to the src/ directory")
end

# `include` creates a real module binding, so GPUIntersection's `using ..Precision`
# resolves here regardless of how a parent module imported it.
include(locate("precision.jl"));        using .Precision
include(locate("gpu_intersection.jl")); using .GPUIntersection

# --------------------------------------------------------------------------- #
#  exact integer linear algebra
# --------------------------------------------------------------------------- #

"""
Exact determinant by Bareiss fraction-free elimination, in place. Every division
is exact by the Bareiss identity, so this stays in the integers. `O(n³)` and
allocation-free — a recursive cofactor expansion would allocate at every level,
which is what made the first version of this file unusably slow in 6D.
"""
function det_bareiss!(A::AbstractMatrix{Int128})
    n = size(A, 1)
    n == 0 && return Int128(1)
    sgn = 1
    prev = Int128(1)
    for k in 1:(n-1)
        if A[k, k] == 0
            piv = 0
            for r in (k+1):n
                if A[r, k] != 0
                    piv = r
                    break
                end
            end
            piv == 0 && return Int128(0)
            for c in 1:n
                A[k, c], A[piv, c] = A[piv, c], A[k, c]
            end
            sgn = -sgn
        end
        for i in (k+1):n, j in (k+1):n
            A[i, j] = (A[i, j] * A[k, k] - A[i, k] * A[k, j]) ÷ prev
        end
        prev = A[k, k]
    end
    return sgn * A[n, n]
end

# --------------------------------------------------------------------------- #
#  synthetic workloads resembling the real one
# --------------------------------------------------------------------------- #

"""
Lattice points of the box `[0,w]^d`, then `n` random `(d+1)`-subsets that are
`unimodular` (|det| = 1, the default `UniTriSat` setting) or merely
non-degenerate. The unimodular restriction matters for tuning: it changes the
geometry, hence how often the early exit fires.
"""
function make_workload(d::Int, n::Int, w::Int, seed::Int; unimodular::Bool = true)
    rng = MersenneTwister(seed)
    pts = collect(Iterators.product(ntuple(_ -> 0:w, d)...))
    P = Matrix{Int}(undef, length(pts), d)
    for (i, p) in enumerate(pts), j in 1:d
        P[i, j] = p[j]
    end
    np = size(P, 1)
    np >= d + 1 || error("box [0,$w]^$d has too few lattice points")
    S = Vector{NTuple{d + 1, Int}}()
    M = Matrix{Int128}(undef, d, d)
    guard = 0
    while length(S) < n && guard < 4000 * n
        guard += 1
        idx = ntuple(_ -> rand(rng, 1:np), d + 1)
        length(unique(idx)) == d + 1 || continue
        @inbounds for r in 1:d, j in 1:d
            M[r, j] = Int128(P[idx[r+1], j] - P[idx[1], j])
        end
        dt = det_bareiss!(M)          # exact: a float det would misclassify
        if unimodular ? abs(dt) == 1 : dt != 0
            push!(S, idx)
        end
    end
    isempty(S) && error("could not sample any simplex for d=$d w=$w")
    return P, S
end

const CACHE = Dict{Tuple{Int,Int,Int,Int,Bool}, Any}()
workload(d, n, w, seed, uni) =
    get!(() -> make_workload(d, n, w, seed; unimodular = uni),
         CACHE, (d, n, w, seed, uni))

# --------------------------------------------------------------------------- #
#  timing one configuration
# --------------------------------------------------------------------------- #

"""
Time the full host call (record building, launches, read-back) for one
configuration, reporting the minimum of `reps` runs after a warm-up. The
warm-up absorbs kernel compilation, which happens once per distinct
configuration.
"""
function bench(d, T, c::GPUIntersection.TuneConfig, n, w, seed, uni, reps)
    P, S = workload(d, n, w, seed, uni)
    # Use the module's own predicate, not a private reimplementation: config_for
    # silently shrinks anything `usable` rejects (including tile^2 < threads),
    # which would benchmark a different configuration than the one requested.
    GPUIntersection.usable(d, T, c) || return (nothing,
        "unusable config (shared=$(GPUIntersection.shared_bytes(d, T, c)) bytes, " *
        "threads=$(c.threads), tile=$(c.tile))")

    key = (d, T)
    saved = get(GPUIntersection.TUNING, key, nothing)
    GPUIntersection.TUNING[key] = c
    try
        hits = length(GPUIntersection.get_intersecting_pairs_gpu(P, S; eltype = T))
        best = Inf
        for _ in 1:reps
            CUDA.synchronize()
            t = time_ns()
            GPUIntersection.get_intersecting_pairs_gpu(P, S; eltype = T)
            CUDA.synchronize()
            best = min(best, (time_ns() - t) / 1e9)
        end
        npairs = length(S) * (length(S) - 1) ÷ 2
        return ((npairs / best / 1e9, best * 1e3, npairs, hits), nothing)
    catch err
        return (nothing, sprint(showerror, err))
    finally
        saved === nothing ? delete!(GPUIntersection.TUNING, key) :
                            (GPUIntersection.TUNING[key] = saved)
    end
end

# --------------------------------------------------------------------------- #
#  independent CPU reference, for `verify`
# --------------------------------------------------------------------------- #

"""
Separating-axis test in `Int128` using the complete family set, written
straightforwardly with Bareiss determinants — deliberately a different
implementation from the generated subset-DP kernel, so that agreement is
evidence rather than tautology.
"""
function cpu_intersects(V1::Matrix{Int128}, V2::Matrix{Int128}, d::Int)
    K = d + 1
    vs    = Matrix{Int128}(undef, max(d - 1, 1), d)
    minor = Matrix{Int128}(undef, max(d - 1, 1), max(d - 1, 1))
    nv    = Vector{Int128}(undef, d)

    dotv(V, r) = begin
        acc = Int128(0)
        for j in 1:d
            acc += V[r, j] * nv[j]
        end
        acc
    end

    for p in 0:(d-1)
        q = d - 1 - p
        fa = p == 0 ? [[1]] : GPUIntersection.subsets(K, p + 1)
        fb = q == 0 ? [[1]] : GPUIntersection.subsets(K, q + 1)
        for A in fa, B in fb
            for i in 1:p, j in 1:d
                vs[i, j] = V1[A[i+1], j] - V1[A[1], j]
            end
            for i in 1:q, j in 1:d
                vs[p+i, j] = V2[B[i+1], j] - V2[B[1], j]
            end
            allzero = true
            for c in 1:d
                jj = 0
                for j in 1:d
                    j == c && continue
                    jj += 1
                    for i in 1:(d-1)
                        minor[i, jj] = vs[i, j]
                    end
                end
                v = det_bareiss!(view(minor, 1:(d-1), 1:(d-1)))
                nv[c] = isodd(c) ? v : -v
                nv[c] == 0 || (allzero = false)
            end
            allzero && continue
            lo1 = hi1 = dotv(V1, 1)
            for r in 2:K
                t = dotv(V1, r)
                t < lo1 && (lo1 = t)
                t > hi1 && (hi1 = t)
            end
            lo2 = hi2 = dotv(V2, 1)
            for r in 2:K
                t = dotv(V2, r)
                t < lo2 && (lo2 = t)
                t > hi2 && (hi2 = t)
            end
            (hi1 <= lo2 || hi2 <= lo1) && return false
        end
    end
    return true
end

"""Vertex matrix of one simplex as `Int128`."""
function vmat(P::Matrix{Int}, tup::NTuple{K, Int}, d::Int) where {K}
    V = Matrix{Int128}(undef, K, d)
    @inbounds for r in 1:K, j in 1:d
        V[r, j] = Int128(P[tup[r], j])
    end
    return V
end

function verify(d, n, w, seed)
    P, S = make_workload(d, n, w, seed)
    got = Set{Tuple{Int,Int}}()
    for cl in GPUIntersection.get_intersecting_pairs_gpu(P, S)
        push!(got, (-cl[1], -cl[2]))
    end
    want = Set{Tuple{Int,Int}}()
    V = [vmat(P, t, d) for t in S]
    for i in 1:length(S), j in (i+1):length(S)
        cpu_intersects(V[i], V[j], d) && push!(want, (i, j))
    end
    miss = length(setdiff(want, got))
    extra = length(setdiff(got, want))
    return (length(S), length(want), miss, extra)
end

# --------------------------------------------------------------------------- #
#  protocol
# --------------------------------------------------------------------------- #

parse_kv(toks) = Dict(String(k) => String(v) for (k, v) in
                      (split(t, '='; limit = 2) for t in toks if occursin('=', t)))
geti(kv, k, dflt) = parse(Int, get(kv, k, string(dflt)))
getb(kv, k, dflt) = get(kv, k, dflt ? "1" : "0") == "1"

function main()
    # the worker sets every configuration explicitly, so the in-process
    # autotuner must stay out of the way
    ENV["UNITRISAT_AUTOTUNE"] = "0"
    CUDA.functional() || (println("fail CUDA not functional"); return)
    dev = CUDA.device()
    cap = CUDA.capability(dev)
    println("ready name=$(replace(CUDA.name(dev), ' ' => '_')) ",
            "sm=$(cap.major)$(cap.minor) ",
            "shared=$(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)) ",
            "maxthreads=$(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)) ",
            "sms=$(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)) ",
            "mem=$(CUDA.totalmem(dev))")
    flush(stdout)

    for line in eachline(stdin)
        toks = split(strip(line))
        isempty(toks) && continue
        cmd = toks[1]
        kv = parse_kv(toks[2:end])
        try
            if cmd == "quit"
                break
            elseif cmd == "bench"
                d = geti(kv, "d", 3)
                T = get(kv, "T", "Int64") == "Int32" ? Int32 : Int64
                c = GPUIntersection.TuneConfig(
                        geti(kv, "threads", 256), geti(kv, "tile", 32),
                        geti(kv, "unroll", 2), getb(kv, "aabb", true),
                        getb(kv, "stab", true))
                res, err = bench(d, T, c, geti(kv, "n", 3000), geti(kv, "w", 3),
                                 geti(kv, "seed", 1), getb(kv, "uni", true),
                                 geti(kv, "reps", 3))
                if res === nothing
                    println("fail $err")
                else
                    g, ms, np, hits = res
                    println(@sprintf("ok gpairs=%.6f ms=%.4f pairs=%d hits=%d",
                                     g, ms, np, hits))
                end
            elseif cmd == "verify"
                ns, nw, miss, extra = verify(geti(kv, "d", 3), geti(kv, "n", 60),
                                             geti(kv, "w", 3), geti(kv, "seed", 1))
                println("ok simplices=$ns expected=$nw missing=$miss extra=$extra")
            else
                println("fail unknown command $cmd")
            end
        catch err
            println("fail ", replace(sprint(showerror, err), '\n' => ' '))
        end
        flush(stdout)
    end
end

end # module

TuneHarness.main()

module SubdivisionRegularity

export is_regular, is_flag_triangulation

using Combinatorics
using LinearAlgebra
using Polyhedra
using CDDLib

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

# Index the vertices without materialising the point matrix (is_flag_triangulation
# never looks at the coordinates).
function index_simplices(triangulation::Vector{Matrix{Int}})
    pt_map = Dict{Vector{Int},Int}()
    unique_points = Vector{Vector{Int}}()
    simplices_idx = Vector{Vector{Int}}(undef, length(triangulation))

    for (i, simplex_mat) in enumerate(triangulation)
        nr = size(simplex_mat, 1)
        s_indices = Vector{Int}(undef, nr)
        for r in 1:nr
            pt = @view simplex_mat[r, :]           # hashes like a Vector{Int}
            idx = get(pt_map, pt, 0)
            if idx == 0
                p = collect(pt)
                push!(unique_points, p)
                idx = length(unique_points)
                pt_map[p] = idx
            end
            s_indices[r] = idx
        end
        simplices_idx[i] = sort!(s_indices)
    end

    return unique_points, simplices_idx
end

function standardize_input(triangulation::Vector{Matrix{Int}})
    unique_points, simplices_idx = index_simplices(triangulation)
    dim = isempty(unique_points) ? 0 : length(first(unique_points))
    pts_matrix = Matrix{Int}(undef, dim, length(unique_points))
    for (j, p) in enumerate(unique_points)
        @inbounds pts_matrix[:, j] = p
    end
    return pts_matrix, simplices_idx
end

# ---------------------------------------------------------------------------
# Exact integer determinant (fraction-free / Bareiss)
# ---------------------------------------------------------------------------
#
# All the matrices here are integral (lattice points plus a row of ones), so
# there is no reason to pay for Rational{BigInt} LU with partial pivoting.

function det_bareiss_int(M::AbstractMatrix{BigInt})
    n = size(M, 1)
    n == 0 && return one(BigInt)
    A = copy(M)
    sgn = 1
    prev = one(BigInt)
    @inbounds for k in 1:(n - 1)
        if iszero(A[k, k])
            r = 0
            for i in (k + 1):n
                if !iszero(A[i, k])
                    r = i
                    break
                end
            end
            r == 0 && return zero(BigInt)
            for c in 1:n
                A[k, c], A[r, c] = A[r, c], A[k, c]
            end
            sgn = -sgn
        end
        for i in (k + 1):n, j in (k + 1):n
            A[i, j] = div(A[i, j] * A[k, k] - A[i, k] * A[k, j], prev)
        end
        prev = A[k, k]
    end
    return sgn * A[n, n]
end

# Divide a constraint row by the gcd of its entries.  The feasibility of
# {w : lambda_i . w > 0 for all i} is invariant under positive rescaling of each
# row, so this is safe and makes duplicate walls collapse.
function normalize_row!(v::Vector{BigInt})
    g = zero(BigInt)
    for x in v
        g = gcd(g, x)
    end
    if !iszero(g) && !isone(g)
        for i in eachindex(v)
            v[i] = div(v[i], g)
        end
    end
    return v
end

# ---------------------------------------------------------------------------
# Regularity
# ---------------------------------------------------------------------------
#
# A triangulation is a regular subdivision iff its secondary cone has non-empty
# interior, and the secondary cone is cut out by the local folding conditions at
# the interior walls (DLRS, Triangulations, Thm. 5.2.11).  For a wall
# F = s1 & s2 with s1 = F + {p}, s2 = F + {q}, the circuit supported on
# F + {p, q} gives the affine dependence lambda, and the condition is
# lambda . w > 0 with lambda normalised so that lambda_p and lambda_q (which
# always share a sign) point the same way at every wall.
#
# Normalisation used below: multiplying the cofactor vector by sign(det s1)
# forces sign(lambda_q) = (-1)^dim at *every* wall, i.e. a uniform convention.
# For odd dim that is the concave rather than the convex convention, which is
# harmless -- regularity is invariant under w -> -w.

#
# `strict = false` (the default) reproduces the original control flow exactly:
# malformed walls and degenerate simplices are skipped silently.  `strict = true`
# turns each of those into an error instead -- they are cases where a skipped
# constraint can turn a non-regular triangulation into a `true`.
function is_regular(triangulation::Vector{Matrix{Int}}; strict::Bool = false)
    if length(triangulation) <= 1
        return true
    end

    pts_int, simplices = standardize_input(triangulation)

    dim = size(pts_int, 1)
    n_points = size(pts_int, 2)

    if strict
        allunique(simplices) || error("is_regular: the triangulation contains duplicate simplices.")
        for s in simplices
            length(s) == dim + 1 || error(
                "is_regular: found a simplex with $(length(s)) vertices in ambient " *
                "dimension $dim; a full-dimensional pure triangulation is required.")
        end
    end

    # homogenised points as columns
    H = Matrix{BigInt}(undef, dim + 1, n_points)
    @inbounds for j in 1:n_points
        for i in 1:dim
            H[i, j] = pts_int[i, j]
        end
        H[dim + 1, j] = 1
    end

    face_map = Dict{Vector{Int},Vector{Int}}()
    for (s_id, s_indices) in enumerate(simplices)
        for face in combinations(s_indices, dim)
            push!(get!(() -> Int[], face_map, face), s_id)
        end
    end

    rows = Set{Vector{BigInt}}()          # deduplicates proportional walls
    M = Matrix{BigInt}(undef, dim + 1, dim + 1)

    for (face, neighbors) in face_map
        nn = length(neighbors)
        if nn != 2
            strict && nn > 2 && error(
                "is_regular: wall $face is shared by $nn simplices; input is not a triangulation.")
            continue                       # boundary wall
        end

        s1 = simplices[neighbors[1]]
        s2 = simplices[neighbors[2]]
        q_candidates = setdiff(s2, face)
        isempty(q_candidates) && continue
        q_idx = q_candidates[1]

        @inbounds for c in 1:(dim + 1)
            for r in 1:(dim + 1)
                M[r, c] = H[r, s1[c]]
            end
        end
        det_s1 = det_bareiss_int(M)
        if iszero(det_s1)
            strict && error(
                "is_regular: degenerate simplex $(s1) (zero volume); cannot decide regularity.")
            continue
        end
        sign_det = sign(det_s1)

        circuit_indices = vcat(s1, q_idx)
        row_coeffs = zeros(BigInt, n_points)

        for i in 1:(dim + 2)
            c = 0
            @inbounds for t in 1:(dim + 2)
                t == i && continue
                c += 1
                pt_idx = circuit_indices[t]
                for r in 1:(dim + 1)
                    M[r, c] = H[r, pt_idx]
                end
            end
            term_det = det_bareiss_int(M)
            row_coeffs[circuit_indices[i]] += (isodd(i) ? -sign_det : sign_det) * term_det
        end

        push!(rows, normalize_row!(row_coeffs))
    end

    isempty(rows) && return true

    # sum(coeff * w) > 0  <=>  sum(coeff * w) >= 1  (the system is a cone)
    # Ax <= b form: sum(-coeff * w) <= -1
    rowvec = collect(rows)
    n_constraints = length(rowvec)
    A = zeros(Rational{BigInt}, n_constraints, n_points)
    b = fill(Rational{BigInt}(-1), n_constraints)
    for i in 1:n_constraints
        r = rowvec[i]
        @inbounds for j in 1:n_points
            A[i, j] = -Rational{BigInt}(r[j])
        end
    end

    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library(:exact))

    return !isempty(poly)
end

# ---------------------------------------------------------------------------
# Flagness
# ---------------------------------------------------------------------------
#
# A complex is flag iff every clique of its 1-skeleton is a face.  Enumerating
# all maximal cliques (Bron-Kerbosch) is the expensive way to check that.  The
# cheap equivalent:
#
#   Delta is flag  <=>  for every face sigma of Delta and every vertex v
#                       adjacent to all of sigma, sigma + {v} is a face.
#
# (=>) trivial.  (<=) induction on |K| for a clique K: |K| <= 2 is a face
# because the 1-skeleton is generated by the facets; for |K| = m > 2 pick any
# v in K, then K \ {v} is a clique of size m-1, a face by induction, and v is
# adjacent to all of it, so K is a face.
#
# So it suffices to walk the faces of Delta (at most 2^(d+1) per facet) and test
# each common neighbour.  No clique enumeration, no exponential blow-up, and it
# bails out at the first violation instead of materialising every maximal
# clique first.  Vertex sets are bitmasks, so "intersect the neighbourhoods" is
# a handful of ANDs.

@inline _lowbit(x::T) where {T<:Unsigned} = x & (~x + one(T))

function _is_flag_mask(simplices::Vector{Vector{Int}}, n::Int, ::Type{M}) where {M<:Unsigned}
    adj = zeros(M, n)
    facets = Vector{M}(undef, length(simplices))

    for (t, s) in enumerate(simplices)
        m = zero(M)
        for v in s
            m |= one(M) << (v - 1)
        end
        facets[t] = m
        for v in s
            adj[v] |= m & ~(one(M) << (v - 1))
        end
    end

    # every face of the complex, as a bitmask
    faces = Set{M}()
    for f in facets
        sub = f
        while true
            push!(faces, sub)
            sub == zero(M) && break
            sub = (sub - one(M)) & f
        end
    end

    for sigma in faces
        count_ones(sigma) >= 2 || continue        # size 0 and 1 always extend

        rest = sigma
        v = trailing_zeros(rest) + 1
        common = adj[v]
        rest &= rest - one(M)
        while rest != zero(M) && common != zero(M)
            v = trailing_zeros(rest) + 1
            common &= adj[v]
            rest &= rest - one(M)
        end

        ext = common & ~sigma
        while ext != zero(M)
            b = _lowbit(ext)
            (sigma | b) in faces || return false  # a clique that is not a face
            ext ⊻= b
        end
    end

    return true
end

# Same algorithm for vertex counts beyond 128.
function _is_flag_wide(simplices::Vector{Vector{Int}}, n::Int)
    adj = [BitSet() for _ in 1:n]
    for s in simplices, i in eachindex(s), j in eachindex(s)
        i == j || push!(adj[s[i]], s[j])
    end

    faces = Set{Vector{Int}}()
    for s in simplices
        k = length(s)
        for mask in 0:((1 << k) - 1)
            push!(faces, [s[i] for i in 1:k if (mask >> (i - 1)) & 1 == 1])
        end
    end

    for sigma in faces
        length(sigma) >= 2 || continue
        common = copy(adj[sigma[1]])
        for t in 2:length(sigma)
            intersect!(common, adj[sigma[t]])
            isempty(common) && break
        end
        for v in common
            v in sigma && continue
            sort!(push!(copy(sigma), v)) in faces || return false
        end
    end

    return true
end

# Checks if a pure simplicial complex (like a subdivision of a polytope)
# is a flag complex. A complex is flag if its minimal non-faces are exclusively edges.
#
# (Checking if a complex is flag is essentially a consequence of its 1-skeleton.)
function is_flag_triangulation(triangulation::Vector{Matrix{Int}})
    if isempty(triangulation)
        error("Flag-checking function was given an empty triangulation?")
    end

    _, simplices_idx = index_simplices(triangulation)

    # In a pure complex, every maximal simplex has the same number of vertices (d + 1).
    target_clique_size = length(first(simplices_idx))
    n = 0
    for s in simplices_idx
        if length(s) != target_clique_size
            error("Triangulation is not pure: found simplices of varying sizes.")
        end
        n = max(n, maximum(s))
    end

    if n <= 64
        return _is_flag_mask(simplices_idx, n, UInt64)
    elseif n <= 128
        return _is_flag_mask(simplices_idx, n, UInt128)
    else
        return _is_flag_wide(simplices_idx, n)
    end
end

# ---------------------------------------------------------------------------
# Bron-Kerbosch with pivoting.  No longer used by is_flag_triangulation; kept
# because it is occasionally useful on its own.  Two fixes relative to the old
# version: Tomita pivoting (maximise |P & N(u)| over P u X) instead of first(P),
# and no eagerly-constructed default Set on every graph lookup.
# ---------------------------------------------------------------------------
function bron_kerbosch!(R::Vector{Int}, P::Set{Int}, X::Set{Int}, graph::Dict{Int,Set{Int}}, cliques::Vector{Vector{Int}})
    if isempty(P) && isempty(X)
        push!(cliques, copy(R))
        return
    end

    empty_nbrs = Set{Int}()
    nbrs(v) = get(graph, v, empty_nbrs)

    pivot = 0
    best = -1
    for u in P
        c = count(w -> w in P, nbrs(u))
        if c > best
            best = c
            pivot = u
        end
    end
    for u in X
        c = count(w -> w in P, nbrs(u))
        if c > best
            best = c
            pivot = u
        end
    end

    for v in collect(setdiff(P, nbrs(pivot)))
        N_v = nbrs(v)
        push!(R, v)
        bron_kerbosch!(R, intersect(P, N_v), intersect(X, N_v), graph, cliques)
        pop!(R)
        delete!(P, v)
        push!(X, v)
    end
end

end

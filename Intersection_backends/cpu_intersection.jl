# cpu_intersection.jl

module CPUIntersection

using LinearAlgebra
using Combinatorics
using Base.Threads

export get_intersecting_pairs_cpu, simplices_intersect_sat_cpu

"""
    _project(vertices::Matrix, axis::Vector) -> Tuple{Real, Real}

Projiziert die Ecken eines Polytops auf eine gegebene Achse und gibt das
minimale und maximale Skalarprodukt zurück.
"""

@inline function _project(vertices::Matrix{Int64}, axis::Vector{Int64})
    n, d = size(vertices)
    @inbounds begin
        # compute first projection
        s = 0
        for j in 1:d
            s += vertices[1, j] * axis[j]
        end
        min_proj = s
        max_proj = s
        # compute the rest
        for i in 2:n
            s = 0
            for j in 1:d
                s += vertices[i, j] * axis[j]
            end
            if s < min_proj
                min_proj = s
            elseif s > max_proj
                max_proj = s
            end
        end
    end
    return min_proj, max_proj
end

"""
    _generalized_cross_product(vectors::Matrix{T}) where T -> Vector{T}

Berechnet das verallgemeinerte Kreuzprodukt für d-1 Vektoren im d-dimensionalen Raum.
Das Ergebnis ist ein Vektor, der orthogonal zu allen Eingabevektoren ist.
Die Eingabe `vectors` ist eine d x (d-1) Matrix, bei der jede Spalte ein Vektor ist.
"""
function _generalized_cross_product(vectors::Matrix{Int64})
    d, n = size(vectors)
    @assert n == d - 1 "Das verallgemeinerte Kreuzprodukt benötigt d-1 Vektoren im d-dimensionalen Raum."
    
    normal = Vector{Int64}(undef, d)
    tmp = Matrix{Int64}(undef, d - 1, d - 1)
    sign = 1

    @views for i in 1:d
        # Computing the minor excluding the ith row
        if i > 1
            tmp[1:(i-1), :] .= vectors[1:(i-1), :]
        end
        if i < d
            tmp[i:(d-1), :] .= vectors[(i+1):d, :]
        end
        sign = -sign
        normal[i] = sign * LinearAlgebra.det_bareiss(tmp)
    end
    return normal
end

struct Simplex
    verts::Matrix{Int64}                 # (num_verts) x d
    facet_normals::Vector{Vector{Int64}} # list of normals (one per facet)
    facet_p0::Vector{Vector{Int64}}      # corresponding p0 for each facet (to orient)
    edges::Vector{Vector{Int64}}         # edge vectors (j>i) stored as Vector{Int64}
    face_edges::Dict{Int, Vector{Vector{Int}}} # maps face_dim => list of edge indices per face
end

"""
    simplices_intersect_sat_cpu(s1_verts::Matrix{T}, s2_verts::Matrix{T}) where T -> Bool

Checks if two d-dimensional simplices (given by their vertex matrices) intersect (on their interior), using the Separating Axis Theorem (SAT).

Returns True if they intersect, Otherwise False.
"""
function simplices_intersect_sat_cpu(s1::Simplex, s2::Simplex)
    s1_verts = s1.verts
    s2_verts = s2.verts
    dim = size(s1_verts, 2)

    function axis_separates(axis::Vector{Int64})
        min1, max1 = _project(s1_verts, axis)
        min2, max2 = _project(s2_verts, axis)
        return max1 <= min2 || max2 <= min1
    end

    # --- Fall 1 & 2: Achsen, die senkrecht zu den Facetten von s1 und s2 stehen ---
    for simplex in (s1, s2)
        for axis in simplex.facet_normals
            if axis_separates(axis)
                return false
            end
        end
    end

    max_k = dim-2      # max dim for s1 face
    max_l = dim-2      # max dim for s2 face
    tmp_combined = zeros(Int64, dim, max_k + max_l)

    # --- Fall 3: Achsen, die aus Seitenflächen beider Simplizes gebildet werden ---
    # Eine Achse wird gebildet, indem man das verallgemeinerte Kreuzprodukt von
    # k Vektoren von einer k-Fläche von s1 und l Vektoren von einer l-Fläche von s2
    # berechnet, wobei k+l = d-1.
    for k in 1:(dim-2)
        l = dim - 1 - k
        for f1_edges in s1.face_edges[k]
            for f2_edges in s2.face_edges[l]
                # combine edges spanning the two faces
                for j in 1:k
                    tmp_combined[:, j] .= s1.edges[f1_edges[j]]
                end
                for j in 1:l
                    tmp_combined[:, k+j] .= s2.edges[f2_edges[j]]
                end
                axis = _generalized_cross_product(tmp_combined[:, 1:(k+l)])
                if any(!iszero, axis) && axis_separates(axis)
                    return false
                end
            end
        end
    end

    # We've enumerated and tested all possible axes but none of them
    # separate. Therefore, the simplices must intersect.
    return true
end

# --- Compute facet normals and edge vectors for a single simplex ---
function compute_simplex_data(verts::Matrix{Int64})
    d = size(verts, 2)
    num_verts = size(verts, 1)  # should equal d+1 for simplices
    @assert num_verts == d + 1 "expected simplex with d+1 vertices"

    facet_normals = Vector{Vector{Int64}}()
    facet_p0 = Vector{Vector{Int64}}()

    # Precompute all edges and index map
    edges = Vector{Vector{Int64}}()
    edge_index = Dict{Tuple{Int,Int}, Int}()
    for i in 1:(num_verts-1)
        for j in (i+1):num_verts
            push!(edges, vec(verts[j, :] .- verts[i, :]))
            edge_index[(i,j)] = length(edges)
        end
    end

    # facets are combinations(1:num_verts, d)
    for facet_indices in combinations(1:num_verts, d)
        p0 = verts[facet_indices[1], :]            # 1 × d view
        # build d-1 spanning vectors as d×(d-1) matrix (columns are vectors)
        span = Matrix{Int64}(undef, d, d-1)
        for j in 2:d
            span[:, j - 1] .= verts[facet_indices[j], :] .- p0
        end
        normal = _generalized_cross_product(span)  # Int64 vector length d
        if all(iszero, normal)
            continue
        end
        # orient to point inward relative to omitted vertex:
        remaining_idx = first(setdiff(1:num_verts, facet_indices))
        p_off_face = verts[remaining_idx, :]
        if dot(normal, p_off_face .- p0) > 0
            normal .= -normal
        end
        push!(facet_normals, normal)
        push!(facet_p0, vec(p0))  # store a copy of p0 (as Vector{Int64})
    end

    # --- precompute faces and exactly k spanning vectors per face ---
    face_edges = Dict{Int, Vector{Vector{Int}}}()
    for k in 1:(d-1)  # face dimension
        face_edges[k] = Vector{Vector{Int}}()
        for face_indices in combinations(1:num_verts, k+1)
            # collect exactly k spanning edges for the generalized cross
            e_idx = Int[]
            p0 = face_indices[1]
            for j in 2:(k+1)
                pj = face_indices[j]
                key = p0 < pj ? (p0,pj) : (pj,p0)
                push!(e_idx, edge_index[key])
            end
            push!(face_edges[k], e_idx)
        end
    end

    return Simplex(verts, facet_normals, facet_p0, edges, face_edges)
end

# Precompute the type conversion and also the generalized cross
# products for each simplex.
function prepare_simplices_cpu(P::Matrix{Rational{BigInt}}, S_indices::Vector)
    num_simplices = length(S_indices)
    simplices = Vector{Simplex}(undef, num_simplices)
    for i in 1:num_simplices
        simplices[i] = compute_simplex_data(convert(Matrix{Int64}, P[collect(S_indices[i]), :]))
    end
    return simplices
end

function get_intersecting_pairs_cpu_generic(P::Matrix{Rational{BigInt}}, S_indices::Vector)
    num_simplices = length(S_indices)

    simplices = prepare_simplices_cpu(P, S_indices)
    total_pairs = div(num_simplices * (num_simplices - 1), 2)

    num_threads = nthreads()

    thread_clauses = [Vector{Vector{Int}}() for _ in 1:nthreads()];
    # Split work evenly among threads
    pairs_per_thread = div(total_pairs + num_threads - 1, num_threads)
    @threads for thread_id in 1:num_threads
        start_idx = (thread_id - 1) * pairs_per_thread + 1
        end_idx = min(thread_id * pairs_per_thread, total_pairs)

        clauses = thread_clauses[thread_id]

        for idx in start_idx:end_idx
            # Compute (i,j) from linear index
            i = Int(floor((1 + sqrt(1 + 8*(idx-1))) / 2))
            acc = div(i*(i-1), 2)
            j = idx - acc + i
            if j > num_simplices
                continue
            end

            t1, t2 = simplices[i], simplices[j]
            if simplices_intersect_sat_cpu(t1, t2)
                push!(clauses, [-i, -j])
            end
        end
    end
    return vcat(thread_clauses...)
end

end # module CPUIntersection

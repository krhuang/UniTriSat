# cpu_intersection.jl

module CPUIntersection

using LinearAlgebra
using Combinatorics

export simplices_intersect_sat_cpu

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
function _generalized_cross_product(vectors::Matrix{T}) where T
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

"""
    simplices_intersect_sat_cpu(s1_verts::Matrix{T}, s2_verts::Matrix{T}) where T -> Bool

Checks if two d-dimensional simplices (given by their vertex matrices) intersect (on their interior), using the Separating Axis Theorem (SAT).

Returns True if they intersect, Otherwise False.
"""
function simplices_intersect_sat_cpu(s1_verts::Matrix{T}, s2_verts::Matrix{T}) where T
    # Patch fix to make simplex vertices Int64's, rather than Rational{BigInt}
    # TODO: fix this better
    s1_verts = convert(Matrix{Int64}, s1_verts) 
    s2_verts = convert(Matrix{Int64}, s2_verts)
    dim = size(s1_verts, 2)
    num_verts = dim + 1

    function axis_separates(axis::Vector{Int64})
        min1, max1 = _project(s1_verts, axis)
        min2, max2 = _project(s2_verts, axis)
        return max1 <= min2 || max2 <= min1
    end

    # --- Fall 1 & 2: Achsen, die senkrecht zu den Facetten von s1 und s2 stehen ---
    for simplex_verts in (s1_verts, s2_verts)
        # Eine Facette wird durch `dim` Ecken definiert
        for facet_indices in combinations(1:num_verts, dim)
            p0 = simplex_verts[facet_indices[1], :]
            # Erzeuge `dim-1` Vektoren, die die Facette aufspannen
            facet_vectors = hcat([simplex_verts[facet_indices[i], :] - p0 for i in 2:dim]...)
            
            axis = _generalized_cross_product(facet_vectors)
            if all(iszero, axis); continue; end

            # Orient the normals so that they point "inwards" in relation to the omitted point.
            # TODO is this necessary?
            remaining_vertex_idx = first(setdiff(1:num_verts, facet_indices))
            p_off_face = simplex_verts[remaining_vertex_idx, :]
            if dot(axis, p_off_face - p0) > 0
                axis = -axis
            end
            if axis_separates(axis)
                return false
            end
        end
    end

    # --- Fall 3: Achsen, die aus Seitenflächen beider Simplizes gebildet werden ---
    # Eine Achse wird gebildet, indem man das verallgemeinerte Kreuzprodukt von
    # k Vektoren von einer k-Fläche von s1 und l Vektoren von einer l-Fläche von s2
    # berechnet, wobei k+l = d-1.
        num_verts = size(s1_verts, 1)

    # Reuse buffers for intermediate results
    max_k = dim - 1
    max_l = dim - 1
    max_vecs = max_k + max_l

    # Assume d = size(s1_verts, 2)
    d = size(s1_verts, 2)
    tmp_f1 = zeros(d, max_k)
    tmp_f2 = zeros(d, max_l)
    combined = zeros(d, max_vecs)

    for k in 1:(dim-1)
        l = dim - 1 - k
        if l < 1
            continue
        end

        for f1_indices in combinations(1:num_verts, k + 1)
            p1_0 = @view s1_verts[f1_indices[1], :]

            # Fill f1_vectors in place
            @views for j in 2:(k+1)
                tmp_f1[:, j-1] .= s1_verts[f1_indices[j], :] .- p1_0
            end

            for f2_indices in combinations(1:num_verts, l + 1)
                p2_0 = @view s2_verts[f2_indices[1], :]

                @views for j in 2:(l+1)
                    tmp_f2[:, j-1] .= s2_verts[f2_indices[j], :] .- p2_0
                end

                combined[:, 1:k] .= @view tmp_f1[:, 1:k]
                combined[:, (k+1):(k+l)] .= @view tmp_f2[:, 1:l]

                axis = _generalized_cross_product(combined[:, 1:(k+l)])

                if any(!iszero, axis)
                    if axis_separates(axis)
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

end # module CPUIntersection

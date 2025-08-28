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
function _project(vertices::Matrix{T}, axis::Vector{T}) where T
    min_proj = dot(vertices[1, :], axis)
    max_proj = min_proj
    for i in 2:size(vertices, 1)
        proj = dot(vertices[i, :], axis)
        if proj < min_proj
            min_proj = proj
        elseif proj > max_proj
            max_proj = proj
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
    d = size(vectors, 1)
    if size(vectors, 2) != d - 1
        error("Das verallgemeinerte Kreuzprodukt benötigt d-1 Vektoren im d-dimensionalen Raum.")
    end
    
    normal = Vector{T}(undef, d)
    for i in 1:d
        # Erstelle eine Submatrix durch Entfernen der i-ten Zeile
        sub_matrix_rows = vcat(1:(i-1), (i+1):d)
        sub_matrix = vectors[sub_matrix_rows, :]
        # Das Vorzeichen ergibt sich aus der Kofaktorentwicklung
        sign = iseven(i + 1) ? 1 : -1
        normal[i] = sign * det(sub_matrix)
    end
    return normal
end

"""
    simplices_intersect_sat_cpu(s1_verts::Matrix{T}, s2_verts::Matrix{T}) where T -> Bool

Überprüft, ob zwei d-dimensionale Simplizes (dargestellt durch ihre Eckenmatrizen)
sich schneiden, unter Verwendung des Separating Axis Theorem (SAT).

Gibt `true` zurück, wenn sie sich schneiden, andernfalls `false`.
"""
function simplices_intersect_sat_cpu(s1_verts::Matrix{T}, s2_verts::Matrix{T}) where T
    dim = size(s1_verts, 2)
    num_verts = dim + 1

    # Die zu testenden Achsen werden in einem Set gespeichert, um Duplikate zu vermeiden
    axes_to_test = Vector{Vector{T}}()

    # --- Fall 1 & 2: Achsen, die senkrecht zu den Facetten von s1 und s2 stehen ---
    for simplex_verts in (s1_verts, s2_verts)
        # Eine Facette wird durch `dim` Ecken definiert
        for facet_indices in combinations(1:num_verts, dim)
            p0 = simplex_verts[facet_indices[1], :]
            # Erzeuge `dim-1` Vektoren, die die Facette aufspannen
            facet_vectors = hcat([simplex_verts[facet_indices[i], :] - p0 for i in 2:dim]...)
            
            axis = _generalized_cross_product(facet_vectors)
            if all(iszero, axis); continue; end

            # Orientiere die Normale so, dass sie nach "innen" zeigt, bezogen auf den ausgelassenen Punkt
            remaining_vertex_idx = first(setdiff(1:num_verts, facet_indices))
            p_off_face = simplex_verts[remaining_vertex_idx, :]
            if dot(axis, p_off_face - p0) > 0
                axis = -axis
            end
            push!(axes_to_test, axis)
        end
    end

    # --- Fall 3: Achsen, die aus Seitenflächen beider Simplizes gebildet werden ---
    # Eine Achse wird gebildet, indem man das verallgemeinerte Kreuzprodukt von
    # k Vektoren von einer k-Fläche von s1 und l Vektoren von einer l-Fläche von s2
    # berechnet, wobei k+l = d-1.
    for k in 1:(dim-1)
        l = dim - 1 - k
        if l < 1; continue; end

        # Iteriere über alle k-Flächen von s1 (definiert durch k+1 Ecken)
        for f1_indices in combinations(1:num_verts, k + 1)
            p1_0 = s1_verts[f1_indices[1], :]
            f1_vectors = [s1_verts[f1_indices[i], :] - p1_0 for i in 2:(k+1)]

            # Iteriere über alle l-Flächen von s2 (definiert durch l+1 Ecken)
            for f2_indices in combinations(1:num_verts, l + 1)
                p2_0 = s2_verts[f2_indices[1], :]
                f2_vectors = [s2_verts[f2_indices[i], :] - p2_0 for i in 2:(l+1)]

                combined_vectors = hcat(f1_vectors..., f2_vectors...)
                axis = _generalized_cross_product(combined_vectors)
                
                if !all(iszero, axis)
                    push!(axes_to_test, axis)
                end
            end
        end
    end
    
    # --- Führe den Projektionstest für alle gesammelten Achsen durch ---
    unique_axes = unique(axes_to_test)
    for axis in unique_axes
        min1, max1 = _project(s1_verts, axis)
        min2, max2 = _project(s2_verts, axis)

        # Wenn die Projektionen sich nicht überlappen, haben wir eine trennende Achse gefunden.
        # Die Simplizes schneiden sich nicht.
        if max1 <= min2 || max2 <= min1
            return false
        end
    end

    # Wenn nach dem Testen aller Achsen keine trennende Achse gefunden wurde,
    # müssen sich die Simplizes schneiden.
    return true
end

end # module CPUIntersection

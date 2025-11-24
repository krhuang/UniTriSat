module BasicComputations

using Combinatorics
using LinearAlgebra
using Polyhedra
using Base.Threads
using StaticArrays
using CDDLib

export all_simplices, internal_faces, lattice_points_via_CDDLib

const CDD_LIB_EXACT = CDDLib.Library(:exact)

# Computes the lattice points of a lattice polytope, via CDDLib backend of Polyhedra
# By default CDDLib uses Floats, but you can configure this

# See the documentation at https://juliapolyhedra.github.io/Polyhedra.jl/stable/polyhedron/

#     "CDDLib.Library creates CDDLib.Polyhedron of type either Float64 or Rational{BigInt}. 
# One can choose the first one using CDDLib.Library(:float) and the second one using 
# CDDLib.Library(:exact), by default it is :float."

# function lattice_points_via_CDDLib(vertices::Matrix{Int})
#     # Convert vertices to exact rationals
#     verts = Rational{BigInt}.(vertices)
#
#     # Build exact polyhedron from vertices
#     poly = polyhedron(vrep(verts), CDDLib.Library(:exact))
#
#     # Compute bounding box
#     min_array = [minimum(vertices[:, i]) for i in 1:size(vertices, 2)]
#     max_array = [maximum(vertices[:, i]) for i in 1:size(vertices, 2)]
#     ranges = [min_array[i]:max_array[i] for i in 1:length(min_array)]
#
#     # Enumerate and collect integer points
#     points_list = Vector{Vector{Int}}()
#
#     for pt_tuple in Iterators.product(ranges...)
#         pt = Rational{BigInt}.(collect(pt_tuple))
#         if in(pt, poly)
#             push!(points_list, collect(Int.(pt_tuple)))
#         end
#     end
#
#     # Convert to a matrix with one point per row
#     if isempty(points_list)
#         return zeros(Int, 0, size(vertices, 2))
#     else
#         return reduce(vcat, [permutedims(p) for p in points_list])
#     end
# end

function lattice_points_via_CDDLib(vertices::Matrix{Int})

    float_threshold = 1e-10

    # 1. Exaktes Polytop mit BigInt-Rationalen erstellen
    # Dies ist notwendig für die exakte Fallback-Prüfung
    verts = Rational{BigInt}.(vertices)
    poly = polyhedron(vrep(verts), CDD_LIB_EXACT)

    # --- NEUER TEIL: H-Repräsentation extrahieren und in Float64 konvertieren ---

    # Holen der H-Repräsentation (Ax <= b) aus dem Polytop
    # Dies geschieht nur einmal und ist der teuerste Teil der Vorbereitung.
    hrep_poly = hrep(poly)
    all_halfspaces = halfspaces(hrep_poly)
    num_hyperplanes = length(all_halfspaces)

    if num_hyperplanes == 0
        # Dieser Fall sollte bei einem V-Polyeder nicht eintreten, aber zur Sicherheit
        @warn "Polytop hat keine Hyperebenen-Repräsentation."
        # Fallback auf die alte (langsame) Methode, nur um sicherzugehen
        return lattice_points_via_CDDLib(vertices)
    end

    # Extrahieren von A (Matrix) und b (Vektor) als exakte Rationale
    A_rational = reduce(vcat, [h.a' for h in all_halfspaces])
    b_rational = [h.β for h in all_halfspaces]

    # Konvertieren von A und b in Float64 für schnelle Berechnungen
    A_float = Float64.(A_rational)
    b_float = Float64.(b_rational)

    # 2. Bounding Box berechnen (wie zuvor)
    min_array = [minimum(vertices[:, i]) for i in 1:size(vertices, 2)]
    max_array = [maximum(vertices[:, i]) for i in 1:size(vertices, 2)]
    ranges = [min_array[i]:max_array[i] for i in 1:length(min_array)]

    # 3. Gitterpunkte sammeln
    points_list = Vector{Vector{Int}}()

    # --- NEUER TEIL: Schwellenwert für den Float-Vergleich ---

    # Dein Vorschlag: 2 * eps(Float64).
    # Dies ist die Maschinengenauigkeit für Zahlen um 1.0.
    # ACHTUNG: Dies ist ein *sehr* kleiner Schwellenwert!
    # Es kann sein, dass wir ihn auf z.B. 1e-12 erhöhen müssen,
    # falls zu viele Punkte fälschlicherweise als "zu nah" eingestuft werden.

    # 4. Optimierte Schleife über alle Punkte in der Bounding Box
    for pt_tuple in Iterators.product(ranges...)
        pt_int_vec = collect(Int.(pt_tuple))
        pt_float = Float64.(pt_int_vec) # Punkt für Float-Berechnung

        use_exact_check = false # Flag für "zu nah"
        is_outside_float = false  # Flag für "sicher außerhalb"

        # Schleife über alle Hyperebenen (A[i,:]*x <= b[i])
        for i in 1:num_hyperplanes
            # Berechne die "Distanz": dist = b[i] - A[i,:] * pt
            # @view ist effizienter, da es keine neue Matrix erstellt
            dot_prod = dot(@view(A_float[i, :]), pt_float)
            dist = b_float[i] - dot_prod

            # abs(dist) <= threshold : Punkt ist zu nah an der Ebene.
            # Wir können keine sichere Float-Aussage treffen.
            if abs(dist) <= float_threshold
                use_exact_check = true
                break # Beende Hyperebenen-Schleife, gehe zur exakten Prüfung

            # dist < 0.0 (und nicht "zu nah"): Punkt ist sicher außerhalb.
            elseif dist < 0.0
                is_outside_float = true
                break # Beende Hyperebenen-Schleife, Punkt wird verworfen
            end

            # Fall dist > float_threshold:
            # Der Punkt ist sicher innerhalb *dieser* Hyperebene.
            # Die Schleife läuft weiter zur nächsten Hyperebene.
        end

        # 5. Entscheidung basierend auf den Flags
        if use_exact_check
            # Fallback: Der Punkt war zu nah, wir müssen exakt rechnen
            pt_rational = Rational{BigInt}.(pt_int_vec)
            if in(pt_rational, poly)
                push!(points_list, pt_int_vec)
            end
        elseif !is_outside_float
            # Sicherer Fall: Der Punkt war nie "zu nah" (use_exact_check=false)
            # und er war nie "sicher außerhalb" (is_outside_float=false).
            # Das bedeutet, er war für *alle* Hyperebenen "sicher innerhalb".
            push!(points_list, pt_int_vec)
        end
        # Der 3. Fall (is_outside_float = true) wird einfach ignoriert.
    end

    # 6. Ergebnisse formatieren (wie zuvor)
    if isempty(points_list)
        return zeros(Int, 0, size(vertices, 2))
    else
        return reduce(vcat, [permutedims(p) for p in points_list])
    end
end

#=
# Access the lattice points of a lattice polytope via Normaliz
function lattice_points_via_Normaliz(vertices::Matrix{Int})
    nverts, d = size(vertices)

    # Lift vertices to d+1 dimension by adding 1
    lifted = hcat(vertices, ones(Int, nverts))

    nmz_vertices = Normaliz.NmzMatrix{Normaliz.NmzRational}(lifted)

    # Construct the cone
    cone = Normaliz.LongLongCone(Dict(:cone => nmz_vertices))

    # Get Hilbert basis (generates all integer points in the cone)
    HB = Normaliz.get_matrix_cone_property(cone, "HilbertBasis")

    # Dehomogenize
    ncols = size(HB,2) - 1
    points = []
    for i in 1:size(HB,1)
        len = size(HB,2)
        if HB[i, len] == 1
            push!(points, [HB[i, j] for j in 1:ncols])
        end
    end

    return [vec[j] for vec in points, j in 1:ncols]
end
=#

# Oscar function for computing lattice points of a convex hull
# Only temporary code...
function lattice_points_via_Oscar(vertices::Matrix{Int})
    polytope = convex_hull(vertices)
    LP = lattice_points(polytope)
    dims = size(LP)
    nrows = dims[1]
    ncols = size(LP[1])[1]
    julia_matrix_LP = [LP[i][j] for i in 1:nrows, j in 1:ncols]
    return julia_matrix_LP
end

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
    simplex_indices = NTuple{d+1,Int}[]
    if n < d + 1
        return simplex_indices
    end

    inds = collect(1:(d+1))             # initial combination
    diffs = Matrix{Int}(undef, d, d)

    while true
        idx_1 = inds[1]
        for j in 1:d
            idx_j = inds[j+1]
            for i in 1:d
                diffs[j, i] = lattice_points[idx_j, i] - lattice_points[idx_1, i]
            end
        end
        det_val = LinearAlgebra.det_bareiss(diffs) # use exact integer determinant
        if det_val != 0 && (!unimodular || abs(det_val) == 1)
            push!(simplex_indices, Tuple(inds))
        end
        next_combination!(inds, n) || break
    end
    return simplex_indices
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

end

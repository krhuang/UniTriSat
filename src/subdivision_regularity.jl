module SubdivisionRegularity

export is_regular

using Combinatorics
using LinearAlgebra
using Polyhedra
using CDDLib

function standardize_input(triangulation::Vector{Matrix{Int}})
    pt_map = Dict{Vector{Int}, Int}()
    unique_points = Vector{Vector{Int}}()
    simplices_idx = Vector{Vector{Int}}(undef, length(triangulation))

    for (i, simplex_mat) in enumerate(triangulation)
        s_indices = Int[]
        for r in 1:size(simplex_mat, 1)
            pt = simplex_mat[r, :]
            if !haskey(pt_map, pt)
                push!(unique_points, pt)
                pt_map[pt] = length(unique_points)
            end
            push!(s_indices, pt_map[pt])
        end
        simplices_idx[i] = sort!(s_indices)
    end

    pts_matrix = hcat(unique_points...)
    return pts_matrix, simplices_idx
end

function is_regular(triangulation::Vector{Matrix{Int}})
    if length(triangulation) <= 1; return true; end

    pts_int, simplices = standardize_input(triangulation)

    pts = Rational{BigInt}.(pts_int)

    dim = size(pts, 1)
    n_points = size(pts, 2)

    face_map = Dict{Vector{Int}, Vector{Int}}()

    for (s_id, s_indices) in enumerate(simplices)
        for face in combinations(s_indices, dim)
            if !haskey(face_map, face)
                face_map[face] = [s_id]
            else
                push!(face_map[face], s_id)
            end
        end
    end

    constraints_row_list = Vector{Vector{Rational{BigInt}}}()

    mat_s1 = zeros(Rational{BigInt}, dim+1, dim+1)
    mat_s1[dim+1, :] .= 1//1

    mat_aug = zeros(Rational{BigInt}, dim+1, dim+1)
    mat_aug[dim+1, :] .= 1//1

    for (face, neighbors) in face_map
        if length(neighbors) == 2
            s1_idx, s2_idx = neighbors
            s1_indices = simplices[s1_idx]
            s2_indices = simplices[s2_idx]

            q_candidates = setdiff(s2_indices, face)
            if isempty(q_candidates); continue; end
            q_idx = q_candidates[1]

            for c in 1:(dim+1)
                col_pt_idx = s1_indices[c]
                mat_s1[1:dim, c] = pts[:, col_pt_idx]
            end

            det_s1 = det(mat_s1)
            if det_s1 == 0; continue; end

            sign_det = sign(det_s1)

            circuit_indices = [s1_indices; q_idx]

            row_coeffs = zeros(Rational{BigInt}, n_points)

            for i in 1:length(circuit_indices)
                current_subset = circuit_indices[1:end .!= i]

                for c in 1:(dim+1)
                    pt_idx = current_subset[c]
                    mat_aug[1:dim, c] = pts[:, pt_idx]
                end

                term_det = det(mat_aug)
                val = (i % 2 == 1 ? -1 : 1) * term_det

                p_idx = circuit_indices[i]
                row_coeffs[p_idx] += val
            end

            final_coeffs = sign_det .* row_coeffs
            push!(constraints_row_list, final_coeffs)
        end
    end

    if isempty(constraints_row_list); return true; end

    # Ax <= b
    # We need sum(coeff * w) > 0
    # Integer equivalent: sum(coeff * w) >= 1
    # Convert to <= form: sum(-coeff * w) <= -1

    n_constraints = length(constraints_row_list)
    A = zeros(Rational{BigInt}, n_constraints, n_points)
    b = Vector{Rational{BigInt}}(undef, n_constraints)

    for i in 1:n_constraints
        A[i, :] = -constraints_row_list[i]
        b[i] = -1//1
    end

    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library(:exact))

    return !isempty(poly)
end
end

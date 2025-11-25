module SubdivisionRegularity
    export is_regular

    using Combinatorics
    using LinearAlgebra
    using JuMP
    using GLPK

    # Helper: Convert raw triangulation to specific point-index format
    function standardize_input(triangulation::Vector{Matrix{Int}})
        # Map unique points (vectors) to integer IDs
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

        # Create a Matrix where columns are points (d x n)
        # We use Float64 for the matrix to ensure det() works,
        # but we round to Int later for the coefficients.
        pts_matrix = hcat(unique_points...)

        return pts_matrix, simplices_idx
    end

    function is_regular(triangulation::Vector{Matrix{Int}})
        if length(triangulation) <= 1; return true; end

        # Preprocessing
        pts, simplices = standardize_input(triangulation)
        dim = size(pts, 1)
        n_points = size(pts, 2)

        # Build Adjacency Graph
        # Map: Sorted Face Indices -> [Simplex IDs]
        face_map = Dict{Vector{Int}, Vector{Int}}()

        # We only care about faces of size `dim`
        for (s_id, s_indices) in enumerate(simplices)
            for face in combinations(s_indices, dim)
                # Face is already sorted from s_indices
                if !haskey(face_map, face)
                    face_map[face] = [s_id]
                else
                    push!(face_map[face], s_id)
                end
            end
        end

        # Setup Linear Program
        # Logic: We look for weights w such that local convexity holds everywhere.
        model = Model(GLPK.Optimizer)
        set_silent(model)

        @variable(model, w[1:n_points])
        @variable(model, eps)

        constraints_count = 0

        # Buffers to reduce allocation inside loop
        # S1 matrix: (d+1) x (d+1) (points + row of ones)
        mat_s1 = zeros(Float64, dim+1, dim+1)
        mat_s1[dim+1, :] .= 1.0

        # Augmented matrix for folding form: (d+1) x (d+1)
        mat_aug = zeros(Float64, dim+1, dim+1)
        mat_aug[dim+1, :] .= 1.0

        # Iterate Internal Faces
        for (face, neighbors) in face_map
            if length(neighbors) == 2
                s1_idx, s2_idx = neighbors

                s1_indices = simplices[s1_idx]
                s2_indices = simplices[s2_idx]

                # Identify q: The point in S2 that is NOT in the face
                q_candidates = setdiff(s2_indices, face)
                if isempty(q_candidates); continue; end # Should not happen in valid triangulation
                q_idx = q_candidates[1]

                for c in 1:(dim+1)
                    col_pt_idx = s1_indices[c]
                    mat_s1[1:dim, c] = pts[:, col_pt_idx]
                end

                sign_det = sign(round(Int, det(mat_s1)))

                if sign_det == 0; continue; end # Degenerate simplex

                # Construct Circuit List: [S1 points; q]
                circuit_indices = [s1_indices; q_idx]

                # Calculate Folding Form coefficients
                # folding_form[k] corresponds to point circuit_indices[k]

                # We build the expression: sum( coeff * w[point_idx] )
                # aff_expr starts at 0
                aff_expr = AffExpr(0.0)

                for i in 1:length(circuit_indices)
                    # Construct matrix excluding row i of the circuit

                    # We select all points in circuit_indices except the i-th one
                    current_subset = circuit_indices[1:end .!= i]

                    # Fill Augmented Matrix
                    for c in 1:(dim+1)
                        pt_idx = current_subset[c]
                        mat_aug[1:dim, c] = pts[:, pt_idx]
                    end

                    term_det = round(Int, det(mat_aug))

                    # (-1)^i * det
                    val = (i % 2 == 1 ? -1 : 1) * term_det

                    # Add to expression
                    add_to_expression!(aff_expr, val, w[circuit_indices[i]])
                end

                # Add Constraint
                # sign_det * folding_form * w >= epsilon
                @constraint(model, sign_det * aff_expr >= eps)
                constraints_count += 1

            end
        end

        if constraints_count == 0; return true; end

        # Solve
        @objective(model, Max, eps)
        @constraint(model, eps <= 1.0) # Boundedness

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # We require strictly positive slack for strict convexity
            return value(eps) > 1e-6
        else
            return false
        end
    end
end

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

# Recursive implementation of the Bron-Kerbosch algorithm with pivoting.
# This algorithm finds all maximal cliques in an undirected graph.
# 
# Arguments:
#   R: The current clique being grown.
#   P: The set of prospective vertices that can be added to the clique.
#   X: The set of vertices already processed (used to prevent duplicate cliques).
#   graph: The adjacency list representation of the 1-skeleton.
#   cliques: The collection where discovered maximal cliques are stored.
function bron_kerbosch!(R::Vector{Int}, P::Set{Int}, X::Set{Int}, graph::Dict{Int, Set{Int}}, cliques::Vector{Vector{Int}})
    # Base case: If there are no more candidates to consider and no excluded 
    # vertices that could form a larger clique, R is a maximal clique.
    if isempty(P) && isempty(X)
        push!(cliques, copy(R))
        return
    end
    
    # Choose a pivot vertex to minimize the number of recursive branches.
    # The pivot is chosen from P or X. We then only need to test vertices in P 
    # that are NOT neighbors of the pivot.
    pivot = isempty(P) ? first(X) : first(P)
    P_without_N_pivot = collect(setdiff(P, get(graph, pivot, Set{Int}())))
    
    for v in P_without_N_pivot
        N_v = get(graph, v, Set{Int}())
        
        push!(R, v)
        # Recursively search with the intersection of candidates/excluded sets 
        # and the neighbors of the current vertex v.
        bron_kerbosch!(R, intersect(P, N_v), intersect(X, N_v), graph, cliques)
        pop!(R)
        
        # Move v from prospective to excluded to prevent finding the same clique again.
        delete!(P, v)
        push!(X, v)
    end
end

# Checks if a pure simplicial complex (like a subdivision of a polytope) 
# is a flag complex. A complex is flag if its minimal non-faces are exclusively edges.

# Checking if a complex is flag is essentially a consequence of its 1-skeleton.
function is_flag_triangulation(triangulation::Vector{Matrix{Int}})
    if isempty(triangulation)
        error("Flag-checking function was given an empty triangulation?")
    end

    # standardize_input is expected to return the mapped points and the list 
    # of simplices represented by their vertex indices.
    pts_int, simplices_idx = standardize_input(triangulation)
    
    # Since the input is a pure subdivision, all provided simplices are guaranteed 
    # to be maximal. We store them in a Set for rapid O(1) membership checking later.
    maximal_simplices = Set{Vector{Int}}(simplices_idx)
    
    # In a pure complex, every maximal simplex has the same number of vertices (d + 1).
    # We read this target size from the first simplex.
    target_clique_size = length(first(simplices_idx)) 

    # Construct the 1-skeleton (adjacency graph) of the triangulation.
    # Vertices are connected by an edge if they appear together in any maximal simplex.
    graph = Dict{Int, Set{Int}}()
    vertices = Set{Int}()
    
    for s in simplices_idx
        # Safety check: Ensure the complex is genuinely pure.
        if length(s) != target_clique_size
            error("Triangulation is not pure: found simplices of varying sizes.")
        end
        
        for i in 1:length(s)
            push!(vertices, s[i])
            if !haskey(graph, s[i])
                graph[s[i]] = Set{Int}()
            end
            
            # Connect the current vertex to all subsequent vertices in the simplex
            for j in (i+1):length(s)
                push!(graph[s[i]], s[j])
                
                # Ensure the graph remains undirected/symmetric
                if !haskey(graph, s[j])
                    graph[s[j]] = Set{Int}()
                end
                push!(graph[s[j]], s[i])
            end
        end
    end

    # Discover all maximal cliques formed by the 1-skeleton.
    cliques = Vector{Vector{Int}}()
    bron_kerbosch!(Int[], vertices, Set{Int}(), graph, cliques)

    # Verify the flag condition: the complex is flag if and only if the clique 
    # complex of its 1-skeleton is identical to the original simplicial complex.
    for clique in cliques
        # If the 1-skeleton forms a maximal clique that doesn't match the expected 
        # dimension of the pure complex, it represents a missing or non-realizable face.
        if length(clique) != target_clique_size
            return false
        end
        
        # Every clique found in the graph must perfectly match one of our original 
        # maximal simplices. If it doesn't, there is a "hollow" void in the complex.
        sort!(clique)
        if !(clique in maximal_simplices)
            return false 
        end
    end

    return true
end

end

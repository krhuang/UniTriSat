module SubdivisionRegularity
	export is_regular
	using Polyhedra, CDDLib, Combinatorics, LinearAlgebra

	# Helper to treat the rows of a matrix as coordinate tuples
	pointset(mat::Matrix{Int}) = Set(Tuple.(eachrow(mat)))

	# Compute all internal faces shared by exactly two simplices
	function internal_faces(triangulation::Vector{Matrix{Int}}, dimension::Int)
	    counts = Dict{Set{NTuple{N,Int}} where N, Int}()
	    for simplex in triangulation
			for face_rows in combinations(1:size(simplex,1), dimension)
				face = simplex[collect(face_rows), :]
				s = pointset(face)
				counts[s] = get(counts, s, 0) + 1
			end
	    end
	    # Faces that occur exactly twice are internal
	    return [reduce(vcat, [collect(x)' for x in f]) for (f, c) in counts if c == 2]
	end

	# Find the two simplices adjacent along a given face
	function adjacent_simplices(triangulation::Vector{Matrix{Int}}, face::Matrix{Int})
	    fset = pointset(face)
	    [s for s in triangulation if issubset(fset, pointset(s))]
	end

	# This is a mess. TODO: Fix it when you get a chance
	function is_regular(triangulation::Vector{Matrix{Int}}, dim::Int)
		if length(triangulation) == 1; return true; end
	    dimension = size(first(triangulation), 2)
	    #points = unique(vcat(triangulation...))
	    points = [x for x in Set(point for simplex in triangulation for point in eachrow(simplex))]
	    n = length(points)
	    internal = internal_faces(triangulation, dimension)
	    # println("Number of internal faces: ", length(internal))

	    A = Vector{Vector{Int}}()  # list of rows

	    for face in internal
			adj = adjacent_simplices(triangulation, face) # TODO: this should just compute a circuit with the first and last points being special
			if length(adj) == 2
				s1, s2 = adj
				p = only(setdiff(pointset(s1), pointset(face)))
				q = only(setdiff(pointset(s2), pointset(face)))
				p_vec = collect(p)'
				q_vec = collect(q)'
				circuit = vcat(s1, q_vec)

				# Constructing the folding form
				# See Definition 5.2.4 "Folding Form" of De Loera, Rambau, Santos - Triangulations: Structures for Algorithms and Applications

				submatrix = hcat(s1, ones(Int, size(s1, 1)))  # hcat adds a column
				sign_det = round(Int, det(submatrix))
				# @assert sign_det = 1 || sign_det = -1 # Input is assumed to be unimodular

				folding_form = zeros(Int, n)
				# Constructing one folding form
				for (i, row_idx) in enumerate(1:size(circuit, 1))
					submatrix = circuit[setdiff(1:size(circuit, 1), [row_idx]), :]  # select rows
					augmented = hcat(submatrix, ones(Int, size(submatrix, 1)))  # add column of ones
					val = (-1)^i * round(Int, det(augmented))
					pt = circuit[row_idx, :]
					idx = findfirst(r -> all(r .== pt), points)
					folding_form[idx] = val
				end
				push!(A, sign_det * folding_form) # sign_det should actually be sign
			end
	    end

	    # Convert list of rows to matrix
	    folding_forms_matrix = reduce(vcat, permutedims.(A))
	    b = zeros(Int, size(folding_forms_matrix, 1))

	    # Build exact polyhedron
	   	# This represents the space we're interested in, with weak inequalities instead of sharp ones
	   	solution_space_closure = polyhedron(hrep(folding_forms_matrix, b), CDDLib.Library(:exact)) 

	    # Check full dimensionality (open feasibility)
	    return dim == n
	end
end

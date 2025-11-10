using Polyhedra, CDDLib, Combinatorics, LinearAlgebra

function internal_faces(triangulation::Vector{Vector{Int}}, dimension::Int)
    counts = Dict{Set{Int}, Int}()
    for simplex in triangulation
        for face in combinations(simplex, dimension)
            s = Set(face)
            counts[s] = get(counts, s, 0) + 1
        end
    end
    return [f for (f, c) in counts if c == 2]
end

function adjacent_simplices(triangulation::Vector{Vector{Int}}, face::Set{Int})
    [s for s in triangulation if issubset(face, s)]
end

function simplex_orientation(points::Matrix{Rational{BigInt}}, simplex::Vector{Int})
    A = hcat(points[simplex, :], ones(Rational{BigInt}, length(simplex)))
    return det(A)
end

# This is a mess. TODO: Fix it when you get a chance
function is_regular_exact(points::Matrix{Rational{BigInt}}, triangulation::Vector{Vector{Int}})
    n, dimension = size(points)
    internal = internal_faces(triangulation, dimension)
    println(size(internal))
    A = Vector{Vector{Rational{BigInt}}}()  # list of rows

    for face in internal
        adj = adjacent_simplices(triangulation, face) # TODO: this should just compute a circuit with the first and last points being special
        if length(adj) == 2
            s1, s2 = adj
            p = only(setdiff(s1, face))
            q = only(setdiff(s2, face))
            circuit = vcat(s1, q)

            # Constructing the folding form
            # See Definition 5.2.4 "Folding Form" of De Loera, Rambau, Santos - Triangulations: Structures for Algorithms and Applications
            #=
            # Sign fixing
            
            o1 = simplex_orientation(points, vcat(collect(face), p)) 
            o2 = simplex_orientation(points, vcat(collect(face), q))
            sgn = sign(o1 * o2)
            if sgn == 0
                continue
            end

            row = zeros(Rational{BigInt}, n)
            row[p] = sgn
            row[q] = -sgn
            push!(A, row)
            =#
            submatrix = points[s1, :]
            # sign/orientation computation for the simplex
            aug = hcat(submatrix, ones(Rational{BigInt}, size(submatrix,1)))  # hcat adds a column
			sign_det = LinearAlgebra.det_bareiss(aug)
            # @assert sign_det = 1 || sign_det = -1 # Input is assumed to be unimodular


            folding_form = zeros(Rational{BigInt}, n)
            # Constructing one folding form
            for (i, point_index) in enumerate(circuit)
            	submatrix = points[setdiff(circuit, point_index), :]            # select rows
				augmented = hcat(submatrix, ones(Rational{BigInt}, size(submatrix, 1)))  # add column of ones
				folding_form[point_index] = (-1)^i * LinearAlgebra.det_bareiss(augmented)
            end
            push!(A, sign_det*folding_form) #sign_det should actually be sign
        end
    end

    # Convert list of rows to matrix
    folding_forms_matrix = reduce(vcat, permutedims.(A))
    b = zeros(Rational{BigInt}, size(folding_forms_matrix, 1))

    # Build exact polyhedron
   	
	P = polyhedron(hrep(folding_forms_matrix, b), CDDLib.Library(:exact))
	println(P)

    # Check full dimensionality (open feasibility)
    return dim(P) == n
end

points2D = Rational{BigInt}.([
    0 0;
    1 0;
    0 1;
    1 1
])
triangulation2D = [
    [1,2,3],
    [2,4,3]
]
println("Regular? ", is_regular_exact(points2D, triangulation2D))

# Mother of all examples

points_nonreg = Rational{BigInt}.([
	0 0; #1
	1 0; #2
	0 1; #3
	2 0; #4
	1 1; #5
	0 2; #6 
	3 0; #7
	2 1; #8
	1 2; #9
	0 3; #10
	4 0; #11
	3 1; #12
	2 2; #13
	1 3; #14
	0 4  #15
])

triangulation_nonreg = [
	[1, 5, 8],
	[5, 8, 9],
	[11, 8, 9],
	[15, 5, 9],
	[1, 3, 5],
	[3, 5, 6],
	[5, 6, 10],
	[6, 10, 15],
	[1, 2, 8],
	[2, 8, 6],
	[8, 6, 7],
	[8, 7, 11],
	[9, 15, 14],
	[9, 14, 13],
	[9, 13, 12],
	[9, 12, 11]
]

println("The mother of all examples is regular?", is_regular_exact(points_nonreg, triangulation_nonreg))
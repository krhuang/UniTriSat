
include("basic_computations.jl")
using .BasicComputations
# Checks if a given subdivision (presumed triangulation, for now) 
# is unimodular via translation to the feasibility of a linear 
# program. 

# is_regular takes as input the lattice points of our polytope, along with a triangulation, given as the simplex indices
function is_regular(lattice_point_configuration::Matrix, simplex_indices::Set) 
	dim = 3
	# Compute the internal faces
	println(internal_faces(P, dim)) # Gives the tuples of regular points
	for internal_face in internal_faces(P, dim)
		for simplex_indices 
			counter = 0
			if issubset(internal_face, simplex_indices) && counter != 2
				# Record the extra point of simplex_indices

				counter++
			end
		end
	end
	
	# closed_feasible_region = hrep() # Create a polytope
	# Check if it has a relative interior
end

is_regular([0 0 0; 3 0 0; 0 3 0; 0 0 3; 1 0 0; 0 1 0; 0 0 1; 2 0 0; 0 2 0; 0 0 2; 1 1 0; 1 0 1; 0 1 1])
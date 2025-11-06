
# Checks if a given subdivision (presumed triangulation, for now) 
# is unimodular via translation to the feasibility of a linear 
# program. 

# is_regular takes as input the lattice points of our polytope, along with a triangulation
function is_regular(P::Matrix, triangulation::Vector) 
	for triangle in triangulation
		for lattice_point in P 
			# Represent lattice_point in terms of the triangle and retrieve an inequality
		end
	end
	closed_feasible_region = hrep() # Create a polytope
	# Check if it has a relative interior
end
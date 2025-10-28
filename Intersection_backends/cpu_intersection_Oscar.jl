module CPUIntersection_Oscar
using Oscar: convex_hull, intersect, lattice_points, dim

function interior_intersection_via_Oscar(s1_vertices::Matrix{Int64}, s2_vertices::Matrix{Int64})
	s1_polytope = convex_hull(s1_vertices)
	s2_polytope = convex_hull(s2_vertices)

	intersection = intersect(s1_polytope, s2_polytope)
	if dim(intersection) == dim(s1_polytope)
		return true
	end
	return false
end

function get_intersecting_pairs_via_Oscar(P::Matrix{Int}, S_indices::Vector) 
	n = length(S_indices)
    clauses = Vector{Vector{Int}}()
    for i in 1:n-1
    	for j in i+1:n 
    		simplex_1 = P[collect(S_indices[i]), :]
    		simplex_2 = P[collect(S_indices[j]), :]
    		if interior_intersection_via_Oscar(simplex_1, simplex_2)
    			push!(clauses, [-i, -j])
    		end
    	end
    end
    return clauses
end

end #end of module

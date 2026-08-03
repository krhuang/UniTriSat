module UniTriSatOscarExt

using UniTriSat
using Oscar

# Method for a single Oscar polyhedron (e.g. cube(3))
# (should naturally throw an error if a non-integral polytope is inputted)
function UniTriSat.triangulate(polytope::Oscar.Polyhedron; kwargs...)
    # Convert Oscar's matrix object into a standard Julia Matrix{Int}
    vmatrix = Int.(Array(Oscar.matrix(Oscar.vertices(polytope))))
    
    # Forward directly to the Matrix{Int} method
    return UniTriSat.triangulate(vmatrix; kwargs...)
end

# Method for a Vector of Oscar polyhedra
function UniTriSat.triangulate(polytopes::Vector{<:Oscar.Polyhedron}; kwargs...)
    vmatrices = [Matrix{Int}(Oscar.matrix(Oscar.vertices(p))) for p in polytopes]
    return UniTriSat.triangulate(vmatrices; kwargs...)
end

end # module
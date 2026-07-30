module UniTriSatOscarExt

using UniTriSat
using Oscar

# Method for a single Oscar polyhedron (e.g. cube(3))
function UniTriSat.triangulate(polytope::Oscar.Polyhedron; kwargs...)
    # Extract vertices from Oscar polyhedron as an integer matrix
    # (Oscar vertices matrix rows/columns conversion to your vmatrix format)
    vmatrix = Int.(Oscar.matrix(Oscar.vertices(polytope)))
    
    # Forward directly to your Matrix{Int} method
    return UniTriSat.triangulate(vmatrix; kwargs...)
end

# Method for a Vector of Oscar polyhedra
function UniTriSat.triangulate(polytopes::Vector{<:Oscar.Polyhedron}; kwargs...)
    vmatrices = [Int.(Oscar.matrix(Oscar.vertices(p))) for p in polytopes]
    return UniTriSat.triangulate(vmatrices; kwargs...)
end

end # module

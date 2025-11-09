using Polyhedra
using CDDLib

hr = HalfSpace([1, 1], 1) ∩ HalfSpace([1, -1], 0) ∩ HalfSpace([-1, 0], 0)

polyf = polyhedron(hr, CDDLib.Library())
println(typeof(polyf))

poly = polyhedron(hr, CDDLib.Library(:exact))
# get the V-representation and convert to a concrete matrix-based VRep
vr = vrep(poly)                    # returns a subtype of VRepresentation
mvr = MixedMatVRep(vr)             # convert to MixedMatVRep{dim,eltype}

# mvr.V is the matrix whose rows (or columns depending on shape) are the points.
# For the typical MixedMatVRep produced by CDDLib, V is a matrix with one row per point.
println("V matrix (points):")
println(mvr.V)
println(typeof(mvr.V))

new_poly = polyhedron(vr, CDDLib.Library(:exact))
println(new_poly)
my_hrep = hrep(new_poly)
println(my_hrep)
println(typeof(my_hrep))
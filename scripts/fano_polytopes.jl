#!/usr/bin/env julia
#
# A script for trying to handle the smooth reflexive polytopes
# produced by Paffenholz & co.
# See https://polymake.org/polytopes/paffenholz/www/fano.html
# as well as https://polymake.org/polytopes/paffenholz/www/rut.html
#
# STUB: vertices_from_hrep is not implemented. For a working H- to
# V-representation conversion see polytope_representation_conversion.jl,
# which does the same job with exact rational arithmetic via CDDLib.

include(joinpath(@__DIR__, "_setup.jl"))

using Polyhedra, CDDLib

function vertices_from_hrep(equations::Matrix{Int})
    error("vertices_from_hrep is not implemented; " *
          "use polytope_representation_conversion.jl instead")
end

import Pkg
Pkg.activate(".")

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include("../src/helpers.jl")
using .Helpers



terminal_output = "initial, running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Fano Matroid}")
println("-")
triangulate(
    "../Polytopes/fano_matroid_base_polytope",
    terminal_output=terminal_output,
    regular=true,
    solver="cadical",
    check_full_dimensionality=true
    )


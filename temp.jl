import Pkg
Pkg.activate(".")

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include("src/helpers.jl")
using .Helpers



terminal_output = "initial, running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue:OhsugiHibiExample 1}")
println("-")
triangulate(
    "Polytopes/OhsugiHibiExample1",
    terminal_output=terminal_output,
    regular=true,
    solver="cadical",
    incremental_solving=true
    )


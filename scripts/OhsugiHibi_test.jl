#!/usr/bin/env julia
#
# Example 1 of Ohsugi-Hibi.
#
#     julia scripts/OhsugiHibi_test.jl

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include(HELPERS)
using .Helpers

check_source(UniTriSat)

terminal_output = "initial, running, table, final" # initial, running, table, final

infile = polytope("OhsugiHibiExample1")

println("-")
println(styled"{bold, blue:OhsugiHibiExample 1}")
println(infile)
println("-")

triangulate(
    infile,
    terminal_output = terminal_output,
    regular = true,
    solver = "cadical",
)

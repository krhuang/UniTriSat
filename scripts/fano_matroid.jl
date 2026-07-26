#!/usr/bin/env julia
#
# Triangulations of the base polytope of the Fano matroid.
#
#     julia scripts/fano_matroid.jl

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include(HELPERS)
using .Helpers

check_source(UniTriSat)

terminal_output = "initial, running, table, final" # initial, running, table, final

infile = polytope("fano_matroid_base_polytope")

println("-")
println(styled"{bold, blue:Fano Matroid}")
println(infile)
println("-")

triangulate(
    infile,
    terminal_output = terminal_output,
    regular = false,
    log_file = logfile("fano_matroid_run"),
    solver = "cadical",
    # check_full_dimensionality = true,
)

#!/usr/bin/env julia
#
# Regular unimodular triangulations of the big smooth 3-polytopes.
#
#     julia scripts/3d_big_smooth_regular.jl [backend]

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat

include(HELPERS)
using .Helpers

check_source(UniTriSat)

backend = argordefault(1, "cpu")

terminal_output = "initial, running, table, final" # initial, running, table, final

infile = polytope("smooth3polytopes_50processed")

println("-")
println(styled"{bold, blue:Dimension 3, Smooth, regular}")
println(infile)
println("-")

triangulate(
    infile,
    terminal_output = terminal_output,
    # log_file = logfile("3d", "smooth_3d"),
    intersection_backend = backend,
    use_normaliz = false,
    regular = true,
)

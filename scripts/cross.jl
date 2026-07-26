#!/usr/bin/env julia
#
# Triangulations of the d-dimensional cross polytope.
#
#     julia scripts/cross.jl 4 [backend]

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include(HELPERS)
using .Helpers

check_source(UniTriSat)

function cross_polytope_vertices(d::Int)
    return [[j == i ? s : 0 for j in 1:d] for i in 1:d for s in [1, -1]]
end

require_args(1, "julia scripts/cross.jl <d> [backend]")

d       = parse(Int, ARGS[1])
backend = argordefault(2, "cpu")

terminal_output = "initial, running, table, final" # initial, running, table, final

println("-")
println(styled"{bold, blue:Cross Polytope Dimension $(d)}")
println("-")

triangulate(
    polyhedron(vrep(cross_polytope_vertices(d))),
    terminal_output = terminal_output,
    # log_file = logfile("$(d)d", "cross_$(backend)"),
    intersection_backend = backend,
    # solver = "cadical",
    # regular = true,
)

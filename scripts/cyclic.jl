#!/usr/bin/env julia
#
# Triangulations of the cyclic polytope C(n, d) on the moment curve.
#
#     julia scripts/cyclic.jl 4 8 [backend]

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include(HELPERS)
using .Helpers

check_source(UniTriSat)

function cyclic_polytope_vertices(d::Int, n::Int)
    return [[i^j for j in 1:d] for i in 0:(n - 1)]
end

require_args(2, "julia scripts/cyclic.jl <d> <n> [backend]")

d       = parse(Int, ARGS[1])
n       = parse(Int, ARGS[2])
backend = argordefault(3, "cpu")

terminal_output = "initial, running, table, final" # initial, running, table, final

println("-")
println(styled"{bold, blue:Cyclic Polytope Dimension $(d), $(n) Vertices}")
println("-")

triangulate(
    polyhedron(vrep(cyclic_polytope_vertices(d, n))),
    terminal_output = terminal_output,
    # log_file = logfile("$(d)d", "v$(n)_$(backend)"),
    intersection_backend = backend,
    # regular = true,
    incremental_solving = true,
    solver = "cadical",
)

#!/usr/bin/env julia
#
# Triangulations of the d-dimensional 0/1 hypercube.
#
#     julia scripts/cube.jl 4 --all

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

include(HELPERS)
using .Helpers

check_source(UniTriSat)

function hypercube_vertices(d::Int)
    # Vertices of the d-dimensional hypercube (0 to 2^d - 1 in binary).
    return [digits(i, base = 2, pad = d) for i in 0:(2^d - 1)]
end

s = ArgParseSettings()
@add_arg_table s begin
    "n"
        help = "dimension of the cube"
        arg_type = Int
        required = true
    "--regular"
        help = "find regular triangulations"
        action = :store_true
    "--all"
        help = "find all triangulations"
        action = :store_true
    "--backend"
        help = "intersection backend, cpu or gpu"
        arg_type = String
        default = "cpu"
    "--solver"
        help = "sat solver, picosat or cadical"
        arg_type = String
        default = "picosat"
    "--incremental_solving"
        help = "enable incremental solving"
        action = :store_true
end

parsed = parse_args(s)

n        = parsed["n"]
regular  = parsed["regular"]
backend  = parsed["backend"]
solver   = parsed["solver"]
find_all = parsed["all"]            # was `all`, which shadowed Base.all
inc_sol  = parsed["incremental_solving"]

terminal_output = "initial, running, table, final"
reg_str = regular  ? ", regular" : ""
all_str = find_all ? ", all triangulations" : ""

println("-")
println(styled"{bold, blue:Hypercube Dimension $n$reg_str$all_str}")
println("-")

triangulate(
    polyhedron(vrep(hypercube_vertices(n))),
    terminal_output = terminal_output,
    intersection_backend = backend,
    regular = regular,
    solver = solver,
    find_all = find_all,
    enable_parallel = true,
    incremental_solving = inc_sol,
)

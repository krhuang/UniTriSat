#!/usr/bin/env julia
#
# All triangulations of the 2-dimensional standard simplex, scaled by n.
#
#     julia scripts/2d_all.jl 3

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

include(HELPERS)
using .Helpers

check_source(UniTriSat)

s = ArgParseSettings()
@add_arg_table s begin
    "n"
        help = "scale"
        arg_type = Int
        required = true
    "--regular"
        help = "find regular triangulations"
        action = :store_true
    "--solver"
        help = "sat solver, picosat or cadical"
        arg_type = String
        default = "picosat"
end

parsed = parse_args(s)

n       = parsed["n"]
regular = parsed["regular"]
solver  = parsed["solver"]

terminal_output = "initial, running, table, final"
reg_str = regular ? ", regular" : ""

println("-")
println(styled"{bold, blue:2d standard simplex, scaled by $n$reg_str, find all}")
println("-")

triangulate(
    [n 0; 0 n; 0 0],
    terminal_output = terminal_output,
    regular = regular,
    solver = solver,
    find_all = true,
)

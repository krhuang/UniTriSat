import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StyledStrings
using Printf
using UniTriSat

using ArgParse
include("../src/helpers.jl")
using .Helpers

vols = vcat(["$i" for i in 1:33], ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

# Argument Parsing Setup (Stil von test.jl)
s = ArgParseSettings()
@add_arg_table s begin
    "d"
        help = "dimension"
        arg_type = Int
        required = true
    "n"
        help = "volume"
        arg_type = Int
        required = true
    "--backend"
        help = "intersection backend, cpu or gpu"
        arg_type = String
        default = "cpu"
    "--solver"
        help = "sat solver, picosat or cadical"
        arg_type = String
        default = "picosat"
    "--regular"
        help = "find regular triangulations"
        action = :store_true
    "--incremental"
        help = "use incremental solving (cadical only)"
        action = :store_true
    "--parallel-solving"
        help = "use parallel solving"
        action = :store_true
    "--all"
        help = "find all triangulations"
        action = :store_true
end
parsed = parse_args(s)
backend = parsed["backend"]
solver = parsed["solver"]
regular = parsed["regular"]
incremental = parsed["incremental"]
enable_parallel = parsed["parallel-solving"]
all = parsed["all"]
d = parsed["d"]
n = parsed["n"]

terminal_output = "initial, running, table, final" #initial, running, table, final


println("-")
println(styled"{bold, blue:Test Dimension $(d), Volume $(n)}")
println("-")
triangulate(
    "../Polytopes/small-lattice-polytopes/data/$(d)-polytopes/v$(n).txt",
    terminal_output=terminal_output,
    regular=regular,
    use_normaliz=false,
    intersection_backend=backend,
    solver=solver,
    return_triangulations = "none",
    incremental_solving=incremental,
    enable_parallel=enable_parallel,
    find_all=all,
    )


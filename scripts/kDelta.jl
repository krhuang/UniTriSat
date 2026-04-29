import Pkg
Pkg.activate(".")

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

if isfile("../src/helpers.jl")
    include("../src/helpers.jl")
    using .Helpers
end

s = ArgParseSettings()
@add_arg_table s begin
    "d"
        help = "dimension of the simplex"
        arg_type = Int
        required = true
    "k"
        help = "scaling factor for the simplex"
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
k = parsed["k"]

# Display Setup
terminal_output = "initial, running, table, final"
reg_str = regular ? ", regular" : ""


triangulate(
    polyhedron(vrep([[i == j ? k : 0 for j in 1:d] for i in 1:d+1])),
    terminal_output=terminal_output,
    intersection_backend=backend,
    regular=regular,
    solver=solver,
    find_all=all,
    incremental_solving=incremental,
    enable_parallel=enable_parallel,
)

import Pkg
Pkg.activate(".")

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

if isfile("src/helpers.jl")
    include("src/helpers.jl")
    using .Helpers
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
    "--backend"
        help = "intersection backend, cpu or gpu"
        arg_type = String
        default = "cpu"
    "--solver"
        help = "sat solver, picosat or cadical"
        arg_type = String
        default = "picosat"
end

parsed = parse_args(s)

n = parsed["n"]
regular = parsed["regular"]
backend = parsed["backend"]
solver = parsed["solver"]

# Display Setup
terminal_output = "initial, running, table, final"
reg_str = regular ? ", regular" : ""


triangulate(
    polyhedron(vrep([[0,0], [n,0], [0,n]])),
    terminal_output=terminal_output,
    intersection_backend=backend,
    regular=regular,
    solver=solver,
    find_all=true,
    # enable_parallel=false
)

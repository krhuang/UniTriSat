import Pkg
Pkg.activate(".")

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

# Import von Helpers, um den Stil von test.jl beizubehalten
# (Geht davon aus, dass die Ordnerstruktur identisch ist)
if isfile("../src/helpers.jl")
    include("../src/helpers.jl")
    using .Helpers
end

function hypercube_vertices(d::Int)
    # Erzeugt Vertices für einen Hyperwürfel der Dimension d (0 bis 2^d - 1 binär)
    return [digits(i, base=2, pad=d) for i in 0:(2^d-1)]
end

# Argument Parsing Setup (Stil von test.jl)
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
        arg_type = Bool
        default = true
end

parsed = parse_args(s)

n = parsed["n"]
regular = parsed["regular"]
backend = parsed["backend"]
solver = parsed["solver"]
all = parsed["all"]
inc_sol = parsed["incremental_solving"]

# Display Setup
terminal_output = "initial, running, table, final"
reg_str = regular ? ", regular" : ""
all_str = all ? ", all triangulations" : ""

println("-")
println(styled"{bold, blue:Hypercube Dimension $n$reg_str$all_str}")
println("-")

# Ausführung der Triangulierung auf lokal generierten Vertices
triangulate(
    polyhedron(vrep(hypercube_vertices(n))),
    terminal_output=terminal_output,
    intersection_backend=backend,
    regular=regular,
    solver=solver,
    find_all=all,
    enable_parallel=true,
    incremental_solving=inc_sol
)

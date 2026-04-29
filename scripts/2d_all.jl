import Pkg
Pkg.activate(".")

using StyledStrings
using UniTriSat
using Polyhedra
using ArgParse

# Import von Helpers, um den Stil von test.jl beizubehalten
# (Geht davon aus, dass die Ordnerstruktur identisch ist)
if isfile("src/helpers.jl")
    include("src/helpers.jl")
    using .Helpers
end

# Argument Parsing Setup (Stil von test.jl)
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

n = parsed["n"]
regular = parsed["regular"]
solver = parsed["solver"]

# Display Setup
terminal_output = "initial, running, table, final"
reg_str = regular ? ", regular" : ""

println("-")
println(styled"{bold, blue:2d standard simplex, scaled by $n$reg_str, find all}")
println("-")

# Ausführung der Triangulierung auf lokal generierten Vertices
triangulate(
    [n 0; 0 n; 0 0],
    terminal_output=terminal_output,
    regular=regular,
    solver=solver,
    find_all=true
)

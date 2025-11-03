using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("helpers.jl")
using .Helpers

vols = ["$i" for i in 1:16]

n = vols[parse(Int, ARGS[1])]

terminal_output = "final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Test Volume $n}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v$(n).txt",
    terminal_output=terminal_output,
    log_file="logs/6d/v$(n)"
    )

using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("helpers.jl")
using .Helpers

vols = vcat(["$i" for i in 1:33], ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

n = vols[parse(Int, ARGS[1])]

terminal_output = "final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Test Volume $n}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v$(n).txt",
    terminal_output=terminal_output,
    log_file="logs/3d/v$(n)"
    )


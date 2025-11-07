using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers

vols = vcat(["$i" for i in 1:33], ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

n = vols[parse(Int, ARGS[2])]
d = ARGS[1]
if length(ARGS) > 2
    backend=ARGS[3]
else
    backend="cpu"
end

terminal_output = "running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Test Dimension $(d), Volume $(n)}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/$(d)-polytopes/v$(n).txt",
    terminal_output=terminal_output,
    log_file="logs/$(d)d/v$(n)_$(backend)",
    intersection_backend=backend
    )


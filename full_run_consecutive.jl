using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers

terminal_output = "running, table, final" #initial, running, table, final

vols = vcat(["$i" for i in 1:33], ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])
ms = [40, 24, 20, 16]

d = parse(Int, ARGS[1])

if length(ARGS) > 1
    backend=ARGS[2]
else
    backend="cpu"
end

lower = parse(Int, ARGS[3])

for n in lower:ms[d-2]
    println("-")
    println(styled"{bold, blue:Dimension $(d), Volume $(n)}")
    println("-")
    results  = triangulate(
        "Polytopes/small-lattice-polytopes/data/$(d)-polytopes/v$(n).txt",
        terminal_output=terminal_output,
        log_file="logs/$(d)d/v$(n)",
        intersection_backend=backend)
end

using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers

terminal_output = "running, table, final" #initial, running, table, final

if length(ARGS) > 0
    backend=ARGS[1]
else
    backend="cpu"
end

println("-")
println(styled"{bold, blue:Bid dataset of smooth 3-Polytopes}")
println("-")
results  = triangulate(
    "Polytopes/smooth3polytopes_50processed",
    terminal_output=terminal_output,
    log_file="logs/3d/big_set_3d",
    intersection_backend=backend
    )


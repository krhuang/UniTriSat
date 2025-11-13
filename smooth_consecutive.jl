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

for d in 4:6
    println("-")
    println(styled"{bold, blue:Dimension $(d), Smooth}")
    println("-")
    triangulate(
        "Polytopes/small-lattice-polytopes/data/smooth/$(d)_polytopes.txt",
        terminal_output=terminal_output,
        log_file="logs/$(d)d/smooth_$(d)d",
        intersection_backend=backend,
        use_normaliz=true
        )
end

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StyledStrings
using Printf
using UniTriSat
include("../src/helpers.jl")
using .Helpers

terminal_output = "initial, running, table, final" #initial, running, table, final

if length(ARGS) > 0
    backend=ARGS[1]
else
    backend="cpu"
end

println("-")
println(styled"{bold, blue:Dimension 3, Smooth, regular}")
println("-")
triangulate(
    "../Polytopes/smooth3polytopes_50processed",
    terminal_output=terminal_output,
    #log_file="logs/$(d)d/smooth_$(d)d",
    intersection_backend=backend,
    use_normaliz=false,
    regular=true
    )

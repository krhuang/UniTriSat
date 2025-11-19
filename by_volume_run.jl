import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StyledStrings
using Printf
using UniTriSat
include("src/helpers.jl")
using .Helpers

vols = vcat(["$i" for i in 1:33], ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

d = ARGS[1]
n = vols[parse(Int, ARGS[2])]
backend = ARGS[3]
if length(ARGS)>3
    regular = true
    if ARGS[4] != "regular"
        @warn("You have passed the unknown option $regular. If you want to search for regular triangulations, please pass 'regular'. Leave empty otherwise.")
        regular = false
    end
else
    regular = false
end

terminal_output = "initial, running, table, final" #initial, running, table, final


println("-")
println(styled"{bold, blue:Test Dimension $(d), Volume $(n) on $(backend)}")
println("-")
triangulate(
    "Polytopes/small-lattice-polytopes/data/$(d)-polytopes/v$(n).txt",
    terminal_output=terminal_output,
#    log_file="logs/$(d)d/v$(n)_$(backend)",
    intersection_backend=backend,
    regular=regular,
    use_normaliz=false,
    return_triangulations = ""
    )


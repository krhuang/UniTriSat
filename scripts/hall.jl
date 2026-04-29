import Pkg
Pkg.activate(".")

using StyledStrings
using Printf
using UniTriSat
using Polyhedra

include("../src/helpers.jl")
using .Helpers

d = parse(Int, ARGS[1])

if length(ARGS) > 1
    backend=ARGS[2]
else
    backend="cpu"
end

file_names_suffixes = ["60", "40", "30", "25", "23", "25", "25"]
file_name = "hall_simplices_$(d)Dim_sum<=$(file_names_suffixes[d-2])"

terminal_output = "initial, running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue: Hall Simplices Dimension $(d), sum <= $(file_names_suffixes[d-2])}")
println("-")
triangulate(
    "../Polytopes/Hall/$(file_name)",
    terminal_output=terminal_output,
    log_file="logs/Hall_$(d)Dim",
    intersection_backend=backend,
    regular=true
    )


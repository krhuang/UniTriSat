using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers


function cross_polytope_vertices(d::Int)
    return [[j == i ? s : 0 for j in 1:d] for i in 1:d for s in [1, -1]]
end

d = parse(Int, ARGS[1])


filename = "Polytopes/Cross/cross_$(d)d"

if !isfile(filename)
    open(filename, "w") do io
        for v in cross_polytope_vertices(d)
            println(io, join(string.(v), " "))
        end
    end
end


if length(ARGS) > 1
    backend=ARGS[2]
else
    backend="cpu"
end

terminal_output = "initial, running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Cross Polytope Dimension $(d)}")
println("-")
results  = triangulate(
    filename,
    terminal_output=terminal_output,
#    log_file="logs/$(d)d/v$(n)_$(backend)",
    intersection_backend=backend,
#    regular=true
    )


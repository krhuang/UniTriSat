using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers


function simplices_product_vertices(d1::Int, d2::Int)
    return [[i^j for j in 1:d] for i in 0:(n-1)]
end

d = parse(Int, ARGS[1])
n = parse(Int, ARGS[2])


filename = "Polytopes/Cyclic/cyclic_$(d)d_$n"

if !isfile(filename)
    open(filename, "w") do io
        for v in cyclic_polytope_vertices(d, n)
            println(io, join(string.(v), " "))
        end
    end
end


if length(ARGS) > 2
    backend=ARGS[3]
else
    backend="cpu"
end

terminal_output = "initial, running, table, final" #initial, running, table, final

println("-")
println(styled"{bold, blue:Product of Simplices of Dimensions $(d1), $(d2) Vertices}")
println("-")
results  = triangulate(
    filename,
    terminal_output=terminal_output,
#    log_file="logs/$(d)d/v$(n)_$(backend)",
    intersection_backend=backend,
#    regular=true
    )


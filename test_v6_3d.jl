
include("Triangulate.jl")
using .Triangulate

print("Test 1, Vol 6 §D. Expect 43 triangulatable polytopes")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output=true,
    validate=false
    )


print("Test 2, Vol 12 §D. Expect 745 triangulatable polytopes")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v12.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output=true,
    validate=false
    )


print("Test 3, Vol 16 §D. Expect 3288 triangulatable polytopes")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output=true,
    validate=false
    )

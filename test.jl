include("Triangulate.jl")
using .Triangulate

print("-")
print("Test 1, Vol 6 3D. Expect 43 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )

print("-")
print("Test 2, Vol 12 3D. Expect 745 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v12.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )


print("-")
print("Test 3, Vol 16 3D. Expect 3288 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )

print("-")
print("Test 4, Vol 10 4D. Expect 618 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/4-polytopes/v10.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )

print("-")
print("Test 5, Vol 10 5D. Expect 841 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/5-polytopes/v10.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )

print("-")
print("Test 6, Vol 10 6D. Expect 959 triangulatable polytopes")
print("-")
print("\n")
triangulate(
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v10.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output="running, final, table",
    validate=false
    )

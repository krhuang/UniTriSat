import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StyledStrings
using Printf
using UniTriSat
include("src/helpers.jl")
using .Helpers

using Profile

function main()
    triangulate(
        "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
        terminal_output="initial, running, table, final",
        intersection_backend="cpu",
        regular=true,
        use_normaliz=false,
        return_triangulations=""
        )
end

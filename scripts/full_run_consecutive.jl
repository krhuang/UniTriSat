#!/usr/bin/env julia
#
# Run consecutively over all volumes of a given dimension.
#
#     julia scripts/full_run_consecutive.jl <d> [backend] [lower]
#
# e.g. julia scripts/full_run_consecutive.jl 4 cpu 1

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat

include(HELPERS)
using .Helpers

check_source(UniTriSat)

terminal_output = "running, table, final" # initial, running, table, final

# Reference list of the available volumes for 3-polytopes. The loop below runs
# over an integer range, so the lettered volumes are not covered by it.
const VOLS = vcat(["$i" for i in 1:33],
                  ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

# Largest volume per dimension, indexed by d - 2, i.e. d = 3, 4, 5, 6.
const MS = [40, 24, 20, 16]

require_args(1, "julia scripts/full_run_consecutive.jl <d> [backend] [lower]")

d = parse(Int, ARGS[1])
3 <= d <= 6 || error("dimension must be between 3 and 6, got $d")

backend = argordefault(2, "cpu")
lower   = parse(Int, argordefault(3, "1"))

for n in lower:MS[d - 2]
    infile = polytope_path("small-lattice-polytopes", "data",
                           "$(d)-polytopes", "v$(n).txt")
    if !isfile(infile)
        @warn "skipping, no such polytope file" volume = n path = infile
        continue
    end

    println("-")
    println(styled"{bold, blue:Dimension $(d), Volume $(n)}")
    println("-")

    triangulate(
        infile,
        terminal_output = terminal_output,
        log_file = logfile("$(d)d", "v$(n)"),
        intersection_backend = backend,
        use_normaliz = true,
    )
end

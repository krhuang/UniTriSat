#!/usr/bin/env julia
#
# Regular unimodular triangulations of the smooth polytopes, dimensions 4 to 6.
#
#     julia scripts/smooth_consecutive.jl [backend]
#
# NOTE: this script previously did `include("Triangulate.jl"); using .Triangulate`
# against a file that no longer exists, and `include("src/helpers.jl")`, which
# resolves relative to scripts/ and therefore never pointed at <root>/src.
# It could not have run in that state. It now calls UniTriSat.triangulate like
# every other script here; check that the keyword semantics still match what
# the old Triangulate module did before trusting the output.

include(joinpath(@__DIR__, "_setup.jl"))

using StyledStrings
using Printf
using UniTriSat

include(HELPERS)
using .Helpers

check_source(UniTriSat)

terminal_output = "running, table, final" # initial, running, table, final

backend = argordefault(1, "cpu")

for d in 4:6
    infile = polytope_path("small-lattice-polytopes", "data", "smooth",
                           "$(d)_polytopes.txt")
    if !isfile(infile)
        @warn "skipping, no such polytope file" dimension = d path = infile
        continue
    end

    println("-")
    println(styled"{bold, blue:Dimension $(d), Smooth, regular}")
    println("-")

    triangulate(
        infile,
        terminal_output = terminal_output,
        log_file = logfile("$(d)d", "smooth_$(d)d"),
        intersection_backend = backend,
        use_normaliz = false,
        regular = true,
    )
end

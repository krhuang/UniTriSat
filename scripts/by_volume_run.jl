#!/usr/bin/env julia
#
# Run UniTriSat on a single polytope file, selected by dimension and volume.
#
#     julia scripts/by_volume_run.jl 4 24

include(joinpath(@__DIR__, "_setup.jl"))

using Printf
using StyledStrings
using ArgParse
using UniTriSat

include(HELPERS)
using .Helpers

check_source(UniTriSat)

# Reference list of the available volumes for 3-polytopes; not used for
# validation, since the admissible set depends on the dimension.
const VOLS = vcat(["$i" for i in 1:33],
                  ["34a", "34b", "35a", "35b", "36a", "36b", "36c"])

s = ArgParseSettings()
@add_arg_table s begin
    "d"
        help = "dimension"
        arg_type = Int
        required = true
    "n"
        help = "volume, e.g. 24 or 34a"
        arg_type = String          # was Int; String admits the lettered volumes
        required = true
    "--data-root"
        help = "root of the small-lattice-polytopes data directory"
        arg_type = String
        default = polytope_path("small-lattice-polytopes", "data")
    "--backend"
        help = "intersection backend, cpu or gpu"
        arg_type = String
        default = "cpu"
    "--solver"
        help = "sat solver, picosat or cadical"
        arg_type = String
        default = "picosat"
    "--regular"
        help = "find regular triangulations"
        action = :store_true
    "--incremental"
        help = "use incremental solving (cadical only)"
        action = :store_true
    "--parallel-solving"
        help = "use parallel solving"
        action = :store_true
    "--all"
        help = "find all triangulations"
        action = :store_true
end

parsed          = parse_args(s)
d               = parsed["d"]
n               = parsed["n"]
data_root       = parsed["data-root"]
backend         = parsed["backend"]
solver          = parsed["solver"]
regular         = parsed["regular"]
incremental     = parsed["incremental"]
enable_parallel = parsed["parallel-solving"]
find_all        = parsed["all"]     # was `all`, which shadowed Base.all

terminal_output = "initial, running, table, final"

polyfile = joinpath(data_root, "$(d)-polytopes", "v$(n).txt")
if !isfile(polyfile)
    # Reuse the diagnostic in _setup.jl for the default layout.
    polytope("small-lattice-polytopes", "data", "$(d)-polytopes", "v$(n).txt")
end

println("-")
println(styled"{bold, blue:Test Dimension $(d), Volume $(n)}")
println(polyfile)
println("-")

triangulate(
    polyfile,
    terminal_output = terminal_output,
    regular = regular,
    use_normaliz = false,
    intersection_backend = backend,
    solver = solver,
    return_triangulations = "none",
    incremental_solving = incremental,
    enable_parallel = enable_parallel,
    find_all = find_all,
)

#!/usr/bin/env julia
#
# by_volume_run.jl -- run UniTriSat on a single polytope file, selected by
# dimension and volume.
#
# Every path is anchored to this file's location (@__DIR__), so the script may
# be invoked from any working directory:
#
#     julia /path/to/UniTriSat/scripts/by_volume_run.jl 4 24
#
# The environment activated is the repository root, i.e. the UniTriSat package
# project itself, which guarantees that `using UniTriSat` loads <root>/src and
# never a snapshot from ~/.julia/packages.

import Pkg

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, ".."))

Pkg.activate(ROOT)
Pkg.instantiate()

using Printf
using StyledStrings
using ArgParse
using UniTriSat

include(joinpath(ROOT, "src", "helpers.jl"))
using .Helpers

# Guard against the failure mode where UniTriSat resolves to a depot copy.
let p = something(pathof(UniTriSat), "")
    startswith(p, ROOT) || @warn(
        "UniTriSat is not loaded from this checkout -- method signatures may be stale",
        loaded_from = p, root = ROOT)
end

# Default location of the small-lattice-polytopes data. Override with --data-root.
const POLYTOPE_DATA = joinpath(ROOT, "Polytopes", "small-lattice-polytopes", "data")

# Reference list of available volumes (3-polytopes); not used for validation
# because the admissible set depends on the dimension.
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
        default = POLYTOPE_DATA
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
    io = IOBuffer()
    println(io, "polytope file not found: ", polyfile)
    dir = dirname(polyfile)
    if isdir(dir)
        entries = sort(readdir(dir))
        println(io, "directory exists with ", length(entries), " entries, first few: ",
                join(first(entries, 10), ", "))
    else
        println(io, "directory does not exist: ", dir)
        println(io, "if the polytope data is a git submodule, initialise it with")
        println(io, "    git -C ", ROOT, " submodule update --init --recursive")
        println(io, "otherwise point the script at the data with --data-root=/path/to/data")
    end
    error(String(take!(io)))
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

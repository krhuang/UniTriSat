# scripts/_setup.jl
#
# Common preamble for every script in scripts/. Include it as the first
# statement of a script:
#
#     include(joinpath(@__DIR__, "_setup.jl"))
#
# It activates the repository environment (the UniTriSat package project, so
# `using UniTriSat` always resolves to <root>/src and never to a snapshot in
# ~/.julia/packages) and provides path helpers anchored to the repository
# rather than to the working directory the script happens to be invoked from.
#
# Set UNITRISAT_NO_INSTANTIATE=1 to skip Pkg.instantiate(), e.g. on a cluster
# where the depot is read-only or already prepared.

import Pkg

const SCRIPTS   = @__DIR__
const ROOT      = normpath(joinpath(SCRIPTS, ".."))
const POLYTOPES = joinpath(ROOT, "Polytopes")
const LOGS      = joinpath(ROOT, "logs")
const HELPERS   = joinpath(ROOT, "src", "helpers.jl")

Pkg.activate(ROOT; io = devnull)
if isempty(get(ENV, "UNITRISAT_NO_INSTANTIATE", ""))
    Pkg.instantiate()
end

"""
    polytope_path(parts...) -> String

Absolute path below `<root>/Polytopes`, without checking for existence.
"""
polytope_path(parts...) = joinpath(POLYTOPES, parts...)

"""
    polytope(parts...) -> String

Absolute path below `<root>/Polytopes`, e.g.

    polytope("small-lattice-polytopes", "data", "4-polytopes", "v24.txt")

Throws with a diagnostic if nothing exists at that location.
"""
function polytope(parts...)
    path = polytope_path(parts...)
    (isfile(path) || isdir(path)) && return path

    io = IOBuffer()
    println(io, "polytope data not found: ", path)
    dir = dirname(path)
    if isdir(dir)
        entries = sort(readdir(dir))
        println(io, "the directory exists and holds ", length(entries),
                " entries, first few: ", join(first(entries, 10), ", "))
    else
        println(io, "the directory does not exist: ", dir)
        println(io, "if the polytope data is a git submodule, initialise it with")
        println(io, "    git -C ", ROOT, " submodule update --init --recursive")
    end
    error(String(take!(io)))
end

"""
    logfile(parts...) -> String

Absolute path below `<root>/logs`, creating the parent directory if needed.
"""
function logfile(parts...)
    path = joinpath(LOGS, parts...)
    mkpath(dirname(path))
    return path
end

"""
    check_source(m::Module)

Warn if `m` was loaded from the package depot rather than from this checkout,
which is the failure mode that makes a script run against stale method
signatures after a fresh clone.
"""
function check_source(m::Module)
    p = something(pathof(m), "")
    startswith(p, ROOT) || @warn(
        "$(nameof(m)) is not loaded from this checkout, signatures may be stale",
        loaded_from = p, root = ROOT)
    return nothing
end

"""
    argordefault(i, default) -> String

`ARGS[i]` if present, otherwise `default`. Keeps the positional-argument
scripts from throwing a BoundsError on optional trailing arguments.
"""
argordefault(i::Int, default) = length(ARGS) >= i ? ARGS[i] : default

"""
    require_args(n, usage)

Throw a usage message unless at least `n` positional arguments were given.
"""
function require_args(n::Int, usage::AbstractString)
    length(ARGS) >= n && return nothing
    error("expected at least $n argument(s)\nusage: $usage")
end

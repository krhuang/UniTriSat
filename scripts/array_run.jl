#!/usr/bin/env julia
#
# One polytope per SLURM array task.
#
# The task list is the sorted enumeration of all v<num><suffix>.txt files under
# <data-root>/<d>-polytopes for the requested dimensions, ordered by
# (d, num, suffix). A single 1-based index therefore selects exactly one
# (dimension, volume) pair, which is what --array needs.
#
#     julia scripts/array_run.jl --count              # N for --array=1-N
#     julia scripts/array_run.jl --list               # the index -> (d, vol) table
#     julia scripts/array_run.jl 17 --solver cadical  # run task 17
#     julia scripts/array_run.jl 5-15                 # run tasks 5 through 15 inclusively
#     julia scripts/array_run.jl --list-kwargs        # what triangulate accepts
#
# With no positional index the script falls back to $SLURM_ARRAY_TASK_ID.
#
# The index depends on --dims and --data-root, so every task in an array must
# be given the same values, or the mapping shifts underneath you. The script
# prints a digest of the task list on startup; grep the job logs for
# "task list" and check that all tasks agree. Safer still, freeze the list once
# with --write-manifest and pass --manifest to the array.

include(joinpath(@__DIR__, "_setup.jl"))

using Printf
using StyledStrings
using ArgParse
using SHA
using UniTriSat

include(HELPERS)
using .Helpers

check_source(UniTriSat)

# Counts per dimension taken from `ms` in full_run_consecutive.jl, d = 3..6.
# For d = 3 this is corroborated by the 40-entry `vols` list in the same file
# (1:33 plus 34a, 34b, 35a, 35b, 36a, 36b, 36c). For d = 4, 5, 6 it is only a
# count and is used purely as a cross-check against what is on disk.
const KNOWN_COUNTS = Dict(3 => 40, 4 => 24, 5 => 20, 6 => 16)

const VOLPAT = r"^v(\d+)([A-Za-z]*)\.txt$"

struct Task
    d::Int
    vol::String
    path::String
end

"""
    volumes_in(dir) -> Vector{Tuple{Int,String,String}}

All `v<num><suffix>.txt` files in `dir` as (num, suffix, filename), sorted
numerically first and then by suffix, so v2 precedes v10 and v34a precedes v34b.
"""
function volumes_in(dir::AbstractString)
    out = Tuple{Int,String,String}[]
    isdir(dir) || return out
    for f in readdir(dir)
        m = match(VOLPAT, f)
        m === nothing && continue
        push!(out, (parse(Int, m.captures[1]), String(m.captures[2]), f))
    end
    sort!(out, by = t -> (t[1], t[2]))
    return out
end

"""
    enumerate_tasks(data_root, dims) -> Vector{Task}

The full task list, ordered by (dimension, volume number, volume suffix).
"""
function enumerate_tasks(data_root::AbstractString, dims::Vector{Int})
    tasks = Task[]
    for d in sort(dims)
        dir = joinpath(data_root, "$(d)-polytopes")
        vs = volumes_in(dir)
        if isempty(vs)
            @warn "no polytope files found for this dimension" dimension = d directory = dir
            continue
        end
        expected = get(KNOWN_COUNTS, d, nothing)
        if expected !== nothing && expected != length(vs)
            @warn("file count differs from the table in full_run_consecutive.jl, " *
                  "the enumeration on disk is what is used",
                  dimension = d, on_disk = length(vs), table = expected)
        end
        for (num, suf, fname) in vs
            push!(tasks, Task(d, "$(num)$(suf)", joinpath(dir, fname)))
        end
    end
    return tasks
end

function read_manifest(path::AbstractString)
    tasks = Task[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s, '\t')
        length(parts) == 3 ||
            error("malformed manifest line (expected d<TAB>vol<TAB>path): $s")
        push!(tasks, Task(parse(Int, parts[1]), String(parts[2]), String(parts[3])))
    end
    return tasks
end

function write_manifest(path::AbstractString, tasks::Vector{Task})
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# index\td\tvol\tpath  (columns after the comment are d, vol, path)")
        for t in tasks
            println(io, t.d, '\t', t.vol, '\t', t.path)
        end
    end
    return path
end

digest(tasks::Vector{Task}) =
    first(bytes2hex(sha256(join(("$(t.d)/$(t.vol)" for t in tasks), '\n'))), 12)

"""
    parse_value(s) -> Any

Convert a --set value to a Julia value: booleans, then Int, then Float64,
then String.
"""
function parse_value(v::AbstractString)
    lv = lowercase(v)
    lv in ("true", "yes", "on")   && return true
    lv in ("false", "no", "off")  && return false
    lv == "nothing"               && return nothing
    i = tryparse(Int, v);      i === nothing || return i
    f = tryparse(Float64, v);  f === nothing || return f
    return String(v)
end

"""
    triangulate_kwargs() -> Union{Set{Symbol},Nothing}

The keyword arguments `triangulate` declares, or `nothing` if some method
slurps arbitrary keywords, in which case no validation is possible.
"""
function triangulate_kwargs()
    names = Set{Symbol}()
    for m in methods(triangulate)
        for k in Base.kwarg_decl(m)
            endswith(String(k), "...") && return nothing
            push!(names, k)
        end
    end
    return names
end

s = ArgParseSettings(description = "Run one polytope per SLURM array index or range.")
@add_arg_table s begin
    "n"
        help = "1-based task index or range (e.g. 5, or 5-15); defaults to \$SLURM_ARRAY_TASK_ID"
        arg_type = String
        required = false
    "--dims"
        help = "comma-separated dimensions forming the task list"
        arg_type = String
        default = "3,4,5,6"
    "--data-root"
        help = "root of the small-lattice-polytopes data directory"
        arg_type = String
        default = polytope_path("small-lattice-polytopes", "data")
    "--manifest"
        help = "read the task list from this file instead of scanning the data root"
        arg_type = String
        default = ""
    "--write-manifest"
        help = "write the enumerated task list to this file and exit"
        arg_type = String
        default = ""
    "--list"
        help = "print the task list and exit"
        action = :store_true
    "--count"
        help = "print the number of tasks and exit"
        action = :store_true
    "--list-kwargs"
        help = "print the keyword arguments triangulate accepts and exit"
        action = :store_true
    "--dry-run"
        help = "resolve the task and print the call without running it"
        action = :store_true
    # ---- triangulate keywords ----
    "--terminal-output"
        help = "any of initial, running, table, final"
        arg_type = String
        default = "initial, running, table, final"
    "--intersection-backend"
        help = "cpu or gpu"
        arg_type = String
        default = "cpu"
    "--solver"
        help = "picosat or cadical"
        arg_type = String
        default = "picosat"
    "--return-triangulations"
        help = "none, one, all"
        arg_type = String
        default = "none"
    "--regular"
        help = "find regular triangulations"
        action = :store_true
    "--use-normaliz"
        help = "use Normaliz for the lattice point computation"
        action = :store_true
    "--incremental-solving"
        help = "use incremental solving (cadical only)"
        action = :store_true
    "--parallel-solving"
        help = "use parallel solving"
        action = :store_true
    "--all"
        help = "find all triangulations"
        action = :store_true
    "--check-full-dimensionality"
        help = "check full dimensionality (passed only when given)"
        action = :store_true
    "--log"
        help = "log file: auto for logs/<d>d/v<vol>, none to disable, or a path"
        arg_type = String
        default = "auto"
    "--set"
        help = "any other triangulate keyword, as key=value; repeatable"
        arg_type = String
        action = :append_arg
end

parsed = parse_args(s)

if parsed["list-kwargs"]
    kws = triangulate_kwargs()
    if kws === nothing
        println("triangulate accepts arbitrary keywords (a method slurps kwargs...)")
    else
        for k in sort(collect(kws), by = String)
            println(k)
        end
    end
    exit(0)
end

dims      = parse.(Int, split(parsed["dims"], ',' ; keepempty = false))
data_root = parsed["data-root"]

tasks = isempty(parsed["manifest"]) ? enumerate_tasks(data_root, dims) :
                                      read_manifest(parsed["manifest"])
isempty(tasks) && error("task list is empty; checked data root $data_root for dimensions $dims")

if !isempty(parsed["write-manifest"])
    write_manifest(parsed["write-manifest"], tasks)
    println("wrote $(length(tasks)) tasks to $(parsed["write-manifest"])")
    exit(0)
end

if parsed["count"]
    println(length(tasks))
    exit(0)
end

if parsed["list"]
    @printf("%-6s %-4s %-6s %s\n", "index", "d", "vol", "path")
    for (i, t) in enumerate(tasks)
        @printf("%-6d %-4d %-6s %s\n", i, t.d, t.vol, t.path)
    end
    println("\n$(length(tasks)) tasks, digest $(digest(tasks))")
    exit(0)
end

# ---- resolve the index/indices ----

n_arg = parsed["n"]
if n_arg === nothing
    env = get(ENV, "SLURM_ARRAY_TASK_ID", "")
    isempty(env) && error("no task index given and SLURM_ARRAY_TASK_ID is unset\n" *
                          "usage: julia scripts/array_run.jl <n> [options]")
    n_arg = env
end

indices = Int[]
if occursin('-', n_arg)
    parts = split(n_arg, '-')
    length(parts) == 2 || error("Invalid range format, expected 'start-end': $n_arg")
    start_idx = parse(Int, parts[1])
    end_idx = parse(Int, parts[2])
    indices = collect(start_idx:end_idx)
else
    push!(indices, parse(Int, n_arg))
end

for idx in indices
    1 <= idx <= length(tasks) ||
        error("task index $idx out of range, the list holds $(length(tasks)) tasks")
end

# ---- assemble the keyword arguments ----

kwargs = Dict{Symbol,Any}(
    :terminal_output       => parsed["terminal-output"],
    :regular               => parsed["regular"],
    :use_normaliz          => parsed["use-normaliz"],
    :intersection_backend  => parsed["intersection-backend"],
    :solver                => parsed["solver"],
    :return_triangulations => parsed["return-triangulations"],
    :incremental_solving   => parsed["incremental-solving"],
    :enable_parallel       => parsed["parallel-solving"],
    :find_all              => parsed["all"],
)

# Only passed when explicitly requested, since it appears in the repository
# solely as a commented-out keyword and may not exist in every version.
parsed["check-full-dimensionality"] && (kwargs[:check_full_dimensionality] = true)

for kv in something(parsed["set"], String[])
    occursin('=', kv) || error("--set expects key=value, got: $kv")
    k, v = split(kv, '=', limit = 2)
    kwargs[Symbol(strip(k))] = parse_value(v)
end

known = triangulate_kwargs()
if known !== nothing
    unknown = sort([String(k) for k in keys(kwargs) if !(k in known)])
    isempty(unknown) ||
        error("triangulate does not accept: " * join(unknown, ", ") *
              "\nrun with --list-kwargs to see the accepted keywords")
end

# ---- run ----

for idx in indices
    task = tasks[idx]
    isfile(task.path) || error("polytope file not found: $(task.path)")

    # Construct the log keyword for each individual task
    log = parsed["log"]
    if log == "auto"
        kwargs[:log_file] = logfile("$(task.d)d", "v$(task.vol)")
    elseif log != "none"
        mkpath(dirname(abspath(log)))
        # Append index if processing multiple tasks to prevent log overwriting
        kwargs[:log_file] = length(indices) > 1 ? "$(log)_$idx" : log
    end

    println("-")
    println(styled"{bold, blue:Task $idx/$(length(tasks)): Dimension $(task.d), Volume $(task.vol)}")
    println(task.path)
    println("task list digest $(digest(tasks)), $(length(tasks)) tasks, dims $(join(dims, \",\"))")
    for k in sort(collect(keys(kwargs)), by = String)
        println("  ", k, " = ", repr(kwargs[k]))
    end
    println("-")

    if parsed["dry-run"]
        println("dry run for index $idx, nothing executed")
        continue
    end

    triangulate(task.path; kwargs...)
end

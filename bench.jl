using StyledStrings
using Printf
using Polyhedra
using LinearAlgebra

include("Triangulate.jl")
using .Triangulate

function create_cube(d::Int, k::Int)
    A_neg = -Matrix{Int}(I, d, d)
    A_pos = Matrix{Int}(I, d, d)
    A = [A_neg; A_pos]
    b_neg = zeros(Int, d)
    b_pos = fill(k, d)
    b = [b_neg; b_pos]
    h = hrep(A, b)
    p = polyhedron(h)
    return p
end

function format_duration(total_seconds::Float64)
    total_seconds_int = floor(Int, total_seconds)
    h = total_seconds_int ÷ 3600
    rem_seconds = total_seconds_int % 3600
    m = rem_seconds ÷ 60
    s = rem_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function format_bytes(b::Real)
    if b > 1024^3
        return @sprintf("%.2f GiB", b / 1024^3)
        elseif b > 1024^2
        return @sprintf("%.2f MiB", b / 1024^2)
        elseif b > 1024
        return @sprintf("%.2f KiB", b / 1024)
    else
        return @sprintf("%d B", b)
    end
end

ResultSummary = NamedTuple{(:d, :k, :lattice_points, :simplices, :total_time, :total_alloc),
                           Tuple{Int, Int, Int, Int, Float64, Int64}}
all_results = ResultSummary[]

for d in (3:6), k in (1:1)
    println(styled"{bold, blue: Cube [0,$k]^$d}")
    cube = create_cube(d,k)

    result = triangulate(
        cube,
        terminal_output="running, table",
        plotter="python plot_triangulation.py"
        )

    total_alloc_run = 0
    for stat in result.step_stats
        println("  $(rpad(stat.name, 40)): $(format_duration(stat.duration_s)), $(format_bytes(stat.alloc_bytes))")
        total_alloc_run += stat.alloc_bytes
    end
    println(styled"{bold: Total time: $(format_duration(result.total_time))}")
    println("-"^50 * "\n")

    push!(all_results, (
        d = d,
        k = k,
        lattice_points = result.num_lattice_points,
        simplices = result.num_simplices_considered,
        total_time = result.total_time,
        total_alloc = total_alloc_run
        ))
end

println("\n" * "="^85)
println(styled"{bold, green: Final Overiew}")
println("="^85)

@printf "%-5s | %-5s | %-16s | %-16s | %-15s | %-15s\n" "d" "k" "Lattice Points" "Simplices" "Total Time" "Total Memory"
println("-"^85)

for summary in all_results
    time_str = format_duration(summary.total_time)
    mem_str = format_bytes(summary.total_alloc)

    @printf "%-5d | %-5d | %-16d | %-16d | %-15s | %-15s\n" summary.d summary.k summary.lattice_points summary.simplices time_str mem_str
end
println("="^85)

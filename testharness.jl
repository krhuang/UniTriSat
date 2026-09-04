# Tests the functionality across different backends; mainly comparing CPU via circuit intersection
import Pkg
Pkg.activate(".")
Pkg.instantiate()
using StyledStrings
using Printf
using UniTriSat
using Test
include("src/subdivision_regularity.jl")
using .SubdivisionRegularity
#= We now include this directly in the GH repository, with permission from Gabriele Balletti
if !isdir("Polytopes/small-lattice-polytopes")
    using Git
    run(`$(git()) clone https://github.com/gabrieleballetti/small-lattice-polytopes Polytopes/small-lattice-polytopes`)
end
=#
# Constant directory where the polytopes are stored. We append to this later 
const DATA_DIR = joinpath(pkgdir(UniTriSat), "Polytopes", "small-lattice-polytopes", "data")

# Track timing for each individual file processed, tagged with which backend/testset it belongs to
const FILE_TIMINGS = Vector{NamedTuple{(:backend, :dim, :vol, :path, :elapsed), Tuple{String, Int, Int, String, Float64}}}()

macro timed_testset(name, expr)
    quote
        local t0 = time()
        @testset $name begin
            $(esc(expr))
        end
        local elapsed = time() - t0
        push!(TIMING_RESULTS, ($name, elapsed))
    end
end

const TIMING_RESULTS = Vector{Tuple{String, Float64}}()

# Wraps triangulate() with timing and records the result under a backend label
function timed_triangulate(backend_label, dim, vol, path; kwargs...)
    t0 = time()
    result = triangulate(path; kwargs...)
    elapsed = time() - t0
    push!(FILE_TIMINGS, (backend=backend_label, dim=dim, vol=vol, path=path, elapsed=elapsed))
    return result
end

total_start = time()

@testset "UniTriSat Full Suite" begin
    #=
    @timed_testset "Standard CPU Backend Polytopes Tests" begin
        test_data = [
            (3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (4, 10, 618, 618),
            (5, 9, 344, 344),
            (6, 7, 97, 97)
        ]
        println("Test suite for CPU backend:")
        for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
            # Anchor path to package root
            println("Dimension: ", dim)
            println("Volume: ", vol)
            println("Expected triangulable: ", exp)
            path = joinpath(DATA_DIR, "$dim-polytopes", "v$vol.txt")
            result = timed_triangulate("cpu", dim, vol, path;
                        terminal_output="initial,running,table,final",
                        intersection_backend="cpu",
                        return_triangulations="",
                        circuit_intersection_clauses=false,
                        solver="picosat",
                        parallel_split_solving=false)
            # The @test macro tracks and reports the success
            @test result.number_triangulatable == exp
        end
    end=#
    @timed_testset "Circuit Backend Polytopes Tests" begin
        test_data = [
            (3,3,5,5)
            #=(3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (4, 10, 618, 618),
            (5, 9, 344, 344),
            (6, 7, 97, 97)=#
        ]
        println("Test suite for Circuits backend:")
        for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
            # Anchor path to package root
            println("Dimension: ", dim)
            println("Volume: ", vol)
            println("Expected triangulable: ", exp)
            path = joinpath(DATA_DIR, "$dim-polytopes", "v$vol.txt")
            result = timed_triangulate("circuit", dim, vol, path;
                        terminal_output="intiial,running,table,final",
                        intersection_backend="cpu",
                        return_triangulations="",
                        circuit_intersection_clauses=true,
                        solver="picosat",
                        parallel_split_solving=false)
            # The @test macro tracks and reports the success
            @test result.number_triangulatable == exp
        end
    end
end

total_elapsed = time() - total_start

# Print time-taken summary
println()
println("="^60)
println("Timing Summary")
println("="^60)

for (name, t) in TIMING_RESULTS
    @printf("%-50s %8.3f s\n", name, t)
end
println("-"^60)
@printf("%-50s %8.3f s\n", "Total", total_elapsed)
println("="^60)

println()
println("Per-file breakdown")
println("="^60)
for backend_label in unique(getfield.(FILE_TIMINGS, :backend))
    println(backend_label, " backend:")
    subset = filter(f -> f.backend == backend_label, FILE_TIMINGS)
    for f in subset
        label = "dim=$(f.dim), vol=$(f.vol)"
        @printf("  %-40s %8.3f s\n", label, f.elapsed)
    end
    backend_total = sum(f.elapsed for f in subset)
    @printf("  %-40s %8.3f s\n", "subtotal", backend_total)
    println()
end
println("="^60)
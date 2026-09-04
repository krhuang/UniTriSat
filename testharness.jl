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

# Track timing for each named testset so we can print a summary at the end
const TIMING_RESULTS = Vector{Tuple{String, Float64}}()

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

total_start = time()

@testset "UniTriSat Full Suite" begin
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
            result = triangulate(path;
                        terminal_output="initial,running,table,final",
                        intersection_backend="cpu",
                        return_triangulations="",
                        circuit_intersection_clauses=false,
                        solver="picosat",
                        parallel_split_solving=false)
            # The @test macro tracks and reports the success
            @test result.number_triangulatable == exp
        end
    end
    @timed_testset "Circuit Backend Polytopes Tests" begin
        test_data = [
            (3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (4, 10, 618, 618),
            (5, 9, 344, 344),
            (6, 7, 97, 97)
        ]
        println("Test suite for Circuits backend:")
        for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
            # Anchor path to package root
            println("Dimension: ", dim)
            println("Volume: ", vol)
            println("Expected triangulable: ", exp)
            path = joinpath(DATA_DIR, "$dim-polytopes", "v$vol.txt")
            result = triangulate(path;
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
println("="^50)
println("Timing Summary")
println("="^50)
for (name, t) in TIMING_RESULTS
    @printf("%-45s %8.3f s\n", name, t)
end
println("-"^50)
@printf("%-45s %8.3f s\n", "Total", total_elapsed)
println("="^50)
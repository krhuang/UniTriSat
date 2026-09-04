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

@testset "UniTriSat Full Suite" begin

    @testset "Standard CPU Backend Polytopes Tests" begin
        test_data = [
            (3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (4, 10, 618, 618),
            (5, 9, 344, 344),
            (6, 7, 94, 94)
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

    @testset "Circuit Backend Polytopes Tests" begin
        test_data = [
            (3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (4, 10, 618, 618),
            (5, 9, 344, 344),
            (6, 7, 94, 94)
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
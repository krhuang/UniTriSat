
using UniTriSat
using Test
using Pkg

include("../src/subdivision_regularity.jl")
using .SubdivisionRegularity

#= We now include this directly in the GH repository, with permission from Gabriele Balletti
if !isdir("Polytopes/small-lattice-polytopes")
    using Git
    run(`$(git()) clone https://github.com/gabrieleballetti/small-lattice-polytopes Polytopes/small-lattice-polytopes`)
end
=#

# mutable struct TriangulationResult
#     solution_simplices::Vector{Vector{Matrix{Int}}}
#     number_of_triangulations_found::Int
#     number_of_regular_triangulations_found::Int
#     minimal_log::String
#     total_time::Float
#     step_stats::Vector{StepStat}
# end

# mutable struct RunResult
#     triangulation_results::Vector{TriangulationResult}
#     number_triangulatable::Int
#     number_regularly_triangulatable::Int
#     total_number_of_triangulations_found::Int
#     total_number_of_regular_triangulations_found::Int
#     total_time::Float
# end

# Constant directory where the polytopes are stored. We append to this later 
const DATA_DIR = joinpath(pkgdir(UniTriSat), "Polytopes", "small-lattice-polytopes", "data")

@testset "UniTriSat Full Suite" begin

    @testset "Polytopes Tests" begin
        test_data = [
            (3, 8, 125, 125),
            (3, 16, 3288, 3288),
            (3, 17, 3784, 3783),
            (3, 19, 7771, 7769),
            (4, 13, 1760, 1760),
            (5, 11, 869, 869),
            (6, 9, 392, 392)
        ]

        for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
            # Anchor path to package root
            path = joinpath(DATA_DIR, "$dim-polytopes", "v$vol.txt")
            
            result = triangulate(path;
                        terminal_output="",
                        intersection_backend="cpu",
                        return_triangulations="",
                        regular=true,
                        solver="picosat",
                        parallel_split_solving=false)

            # The @test macro tracks and reports the success
            @test result.number_triangulatable == exp
            @test result.number_regularly_triangulatable == exp_reg
        end
    end

    @testset "Regularity Check" begin
        triangulation_nonreg = [
            [0 0; 0 1; 1 1], [0 0; 1 1; 2 1], [0 0; 1 0; 2 1],
            [1 0; 2 0; 2 1], [2 0; 3 0; 2 1], [3 0; 4 0; 2 1],
            [4 0; 2 1; 1 2], [4 0; 3 1; 1 2], [0 1; 1 1; 0 2],
            [1 1; 0 2; 1 3], [1 1; 1 3; 0 4], [1 1; 1 2; 0 4],
            [1 1; 2 1; 1 2], [3 1; 1 2; 2 2], [1 2; 2 2; 1 3],
            [1 2; 1 3; 0 4]
        ]
        
        # Test that it is NOT regular
        @test is_regular(triangulation_nonreg) == false
    end
end
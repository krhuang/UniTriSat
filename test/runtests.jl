
using UniTriSat

include("../src/subdivision_regularity.jl")
using .SubdivisionRegularity

if !isdir("Polytopes/small-lattice-polytopes")
    using Git
    run(`$(git()) clone https://github.com/gabrieleballetti/small-lattice-polytopes Polytopes/small-lattice-polytopes`)
end

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

struct TestResult 
    name::String
    pass::Bool
    reason::String
    time::String
end

struct Test
    id::Int
    dim::Int
    vol::Int
    exp::Int
    exp_reg::Int
end

function run_test(test::Test)
    path = "Polytopes/small-lattice-polytopes/data/$(test.dim)-polytopes/v$(test.vol).txt"
    result  = triangulate(
                            path,
                            terminal_output="",
                            intersection_backend="cpu",
                            return_triangulations="",
                            regular=true,
                            solver="picosat",
                            incremental_solving=false,
                            enable_parallel=false)
    num_triangulatable = result.number_triangulatable
    num_reg_triangulatable = result.number_regularly_triangulatable
    if  num_triangulatable == test.exp && (!regular || num_reg_triangulatable == test.exp_reg)
        pass = true
        reason = ""
    else
        reason = ""
        if num_triangulatable != test.exp
            reason *= "Expected $(test.exp) triangulatable, got $num_triangulatable."
        end
        if (regular || num_reg_triangulatable == test.exp_reg)
            reason *= " Expected $(test.exp_reg) regularly triangulatable, got $num_reg_triangulatable."
        end
        @error("$reason")
        pass = false
    end
    return TestResult(name, pass, reason, time)
end
test_data = [   (3, 8, 125, 125),
                (3, 16, 3288, 3288),
                (3, 17, 3784, 3783),
                (3, 19, 7771, 7769),
                (4, 13, 1760, 1760),
                (5, 11, 869, 869),
                (6, 9, 392, 392)]

for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
    test = Test(i, dim, vol, exp, exp_reg)
    res = run_test(test)
end

if true

    # Define triangulation directly as a vector of matrices of integer coordinates
    triangulation_nonreg = [
        [0 0; 0 1; 1 1],
        [0 0; 1 1; 2 1],
        [0 0; 1 0; 2 1],
        [1 0; 2 0; 2 1],
        [2 0; 3 0; 2 1],
        [3 0; 4 0; 2 1],
        [4 0; 2 1; 1 2],
        [4 0; 3 1; 1 2],
        [0 1; 1 1; 0 2],
        [1 1; 0 2; 1 3],
        [1 1; 1 3; 0 4],
        [1 1; 1 2; 0 4],
        [1 1; 2 1; 1 2],
        [3 1; 1 2; 2 2],
        [1 2; 2 2; 1 3],
        [1 2; 1 3; 0 4]
    ]

    elapsed_time = @elapsed begin
        if is_regular(triangulation_nonreg)
            @error("Falsely found that the Mother of all Examples is regular")
        end
    end
end

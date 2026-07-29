import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StyledStrings
using Printf
using UniTriSat

include("src/helpers.jl")
using .Helpers
include("src/subdivision_regularity.jl")
using .SubdivisionRegularity

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
    exp_reg_str = regular ? " and $(test.exp_reg) regularly triangulatable polytopes" : ""
    path = abspath(joinpath(@__DIR__, "Polytopes/small-lattice-polytopes/data/$(test.dim)-polytopes/v$(test.vol).txt")) # Since the files are in the package
    name = "$(test.dim)D Vol $(test.vol)$reg_str"
    println("-")
    println(styled"{bold, blue:Test $(test.id): $name. Expect $(test.exp) triangulatable polytopes$exp_reg_str}")
    println("-")
    result  = triangulate(
                            path,
                            terminal_output=terminal_output,
                            intersection_backend=backend,
                            return_triangulations="none",
                            regular=regular,
                            solver=solver,
                            incremental_solving=incremental,
                            parallel_split_solving=parallel_split_solving)
    num_triangulatable = result.number_triangulatable
    num_reg_triangulatable = result.number_regularly_triangulatable
    time = format_duration(result.total_time)
    if  num_triangulatable == test.exp && (!regular || num_reg_triangulatable == test.exp_reg)
        println(styled"{bold, green:passed} in $time\n")
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
        println(styled"{bold, red:failed}, $reason")
        pass = false
    end
    return TestResult(name, pass, reason, time)
end

using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
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
  "--plot"
    help = "produce plots"
    action = :store_true
  "--incremental"
    help = "use incremental solving (cadical only)"
    action = :store_true
  "--parallel-solving"
    help = "use parallel solving"
    action = :store_true
  "--big"
    help = "run on the 2 big 3D polytopes, but without plotting"
    action = :store_true
end
parsed = parse_args(s)
backend = parsed["backend"]
solver = parsed["solver"]
regular = parsed["regular"]
plot = parsed["plot"]
incremental = parsed["incremental"]
parallel_split_solving = parsed["parallel-solving"]
big = parsed["big"]
reg_str = regular ? ", regular" : ""

test_data = [   (3, 8, 125, 125),
                (3, 16, 3288, 3288),
                (3, 17, 3784, 3783),
                (3, 19, 7771, 7769),
                (4, 13, 1760, 1760),
                (5, 11, 869, 869),
                (6, 9, 392, 392)]


terminal_output = "running, table, final" #initial, running, table, final
test_results = Vector{TestResult}()

if plot || big
    println("-")
    println(styled"{bold, blue:Test two big 3D Polytopes, plot = $plot}")
    println("-")
    triangulate("Polytopes/Big3D", 
                plot=plot,
 #               log_file = "cnf",
                return_triangulations="none",
                intersection_backend=backend,
                terminal_output=terminal_output,
                solver=solver,
                incremental_solving=incremental,
                parallel_split_solving=parallel_split_solving)
end

for (i, (dim, vol, exp, exp_reg)) in enumerate(test_data)
    test = Test(i, dim, vol, exp, exp_reg)
    res = run_test(test)
    push!(test_results, res)
end

if regular
    name = "Regularity of the \"Mother of all Examples\""
    println("-")
    println(styled"{bold, blue:Test $name. Expect non-regular}")
    println("-")

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
        regularity_result = is_regular(triangulation_nonreg)
    end

    if regularity_result
        reason = "falsely found that the Mother of all Examples is regular"
        println(styled"{bold, red:failed}, $reason")
        pass = false
    else
        println(styled"{bold, green:passed} in $(format_duration(elapsed_time)). The Mother of all Examples is not a regular subdivision.\n")
        pass = true
        reason = ""
    end
    push!(test_results, TestResult(name, pass, reason, format_duration(elapsed_time)))
end

# Final summary
for (i, test) in enumerate(test_results)
    print("Test $i: ")
    print(test.name)  # preserves styled formatting
    if test.pass
        println(styled"{bold, green: passed} ($(test.time))")
    else
        println(styled"{bold, red: failed}: $(test.reason) ($(test.time))")
    end
end

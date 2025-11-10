using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers


terminal_output = "running, table, final" #initial, running, table, final

test_results = []
test_num = 1

test_names = []

if length(ARGS) > 0
    backend=ARGS[1]
else
    backend="cpu"
end

println("-")
println(styled"{bold, blue:Test plotting}")
println("-")
results  = triangulate(
    "Polytopes/Big3D",
    terminal_output=terminal_output, intersection_backend=backend, plot=true
    )

test_name = "Vol 6 3D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 43 triangulatable polytopes}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
    terminal_output=terminal_output, intersection_backend=backend
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 43
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1


test_name = "Vol 12 3D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 745 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v12.txt",
    terminal_output=terminal_output
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 745
total_time = sum([result.total_time for result in results])
    if  num_triangulatable == expected_number
        println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

test_name = "Vol 16 3D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 3288 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    terminal_output=terminal_output, intersection_backend=backend
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 3288
total_time = sum([result.total_time for result in results])
    if  num_triangulatable == expected_number
        println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

test_name = "Vol 10 4D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 618 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/4-polytopes/v10.txt",
    terminal_output=terminal_output
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 618
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

test_name = "Vol 10 5D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 841 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/5-polytopes/v10.txt",
    terminal_output=terminal_output
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 841
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

test_name = "Vol 10 6D"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect 959 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v10.txt",
    terminal_output=terminal_output
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 959
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "$(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Regularity of the \"Mother of all Examples\"}. Expect non-regular.")
println("-")

points_nonreg = Rational{BigInt}.([
    0 0; #1
    1 0; #2
    2 0; #3
    3 0; #4
    4 0; #5
    0 1; #6
    1 1; #7
    2 1; #8
    3 1; #9
    0 2; #10
    1 2; #11
    2 2; #12
    0 3; #13
    1 3; #14
    0 4  #15
])

triangulation_nonreg = [
    [1, 6, 7],
    [1, 7, 8],
    [1, 2, 8],
    [2, 3, 8],
    [3, 4, 8],
    [4, 5, 8],
    [5, 8, 11],
    [5, 9, 11],
    [6, 7, 10],
    [7, 10, 13],
    [7, 13, 15],
    [7, 11, 15],
    [7, 8, 11],
    [9, 11, 12],
    [11, 12, 14],
    [11, 14, 15]
]

elapsed_time = @elapsed begin
    regularity_result = is_regular(points_nonreg, triangulation_nonreg)
end

if regularity_result
    println(styled"{bold, red:failed}, falsely found that the Mother of all Examples is regular")
    push!(test_results, "failed: falsely found regular (time $(format_duration(elapsed_time)))")
else
    println(styled"{bold, green:passed} in $(format_duration(elapsed_time)). The Mother of all Examples is not a regular subdivision.\n")
    push!(test_results, "$(format_duration(elapsed_time))")
end
test_num += 1

# Final summary
for (i, res) in enumerate(test_results)
    if !startswith(res, "failed")
        println(styled"Test $(test_names[i]): {bold, green: passed} in $res")
    else
        println(styled"Test $(test_names[i]): {bold, red:$res}")
    end
end
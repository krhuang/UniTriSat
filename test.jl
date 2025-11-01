using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate

function format_duration(total_seconds::Float64)
    total_seconds_int = floor(Int, total_seconds)
    h = total_seconds_int ÷ 3600
    rem_seconds = total_seconds_int % 3600
    m = rem_seconds ÷ 60
    s = rem_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

test_results = []
test_num = 1


println("-")
println(styled"{bold, blue:Test $test_num, Vol 6 3D. Expect 43 triangulatable polytopes}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
    terminal_output="running, table",
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 43
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Vol 12 3D. Expect 745 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v12.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 745
total_time = sum([result.total_time for result in results])
    if  num_triangulatable == expected_number
        println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Vol 16 3D. Expect 3288 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    terminal_output="running, table",
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 3288
total_time = sum([result.total_time for result in results])
    if  num_triangulatable == expected_number
        println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Vol 10 4D. Expect 618 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/4-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 618
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Vol 10 5D. Expect 841 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/5-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 841
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1

println("-")
println(styled"{bold, blue:Test $test_num, Vol 10 6D. Expect 959 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 959
total_time = sum([result.total_time for result in results])
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed} in $(format_duration(total_time))\n")
    push!(test_results, "passed in $(format_duration(total_time))")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end
test_num += 1


for (i,res) in enumerate(test_results)
    if startswith(res, "passed")
        println(styled"Test $i: {bold, green: passed}")
    else
        println(styled"Test $i: {bold, red:$res}")
    end
end

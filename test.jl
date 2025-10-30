using StyledStrings


include("Triangulate.jl")
using .Triangulate


test_results = []

println("-")
println(styled"{bold, blue:Test 1, Vol 6 3D. Expect 43 triangulatable polytopes}")
println("-")
results  = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt",
    terminal_output="running, table",
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 43
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end

println("-")
println(styled"{bold, blue:Test 2, Vol 12 3D. Expect 745 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v12.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 745
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end

println("-")
println(styled"{bold, blue:Test 3, Vol 16 3D. Expect 3288 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    terminal_output="running, table",
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 3288
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end

println("-")
println(styled"{bold, blue:Test 4, Vol 10 4D. Expect 618 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/4-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 618
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end

println("-")
println(styled"{bold, blue:Test 5, Vol 10 5D. Expect 841 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/5-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 841
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end

println("-")
println(styled"{bold, blue:Test 6, Vol 10 6D. Expect 959 triangulatable polytopes}")
println("-")
results = triangulate(
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v10.txt",
    terminal_output="running, table"
    )
num_triangulatable = length([1 for result in results if result.num_solutions_found>0])
expected_number = 959
if  num_triangulatable == expected_number
    println(styled"{bold, green:passed}\n")
    push!(test_results, "passed")
else
    println(styled"{bold, red:failed}, Expected $expected_number, got $num_triangulatable")
    push!(test_results, "failed: Expected $expected_number, got $num_triangulatable")
end


for (i,res) in enumerate(test_results)
    if res == "passed"
        println(styled"Test $i: {bold, green: passed}")
    else
        println(styled"Test $i: {bold, red:$res}")
    end
end

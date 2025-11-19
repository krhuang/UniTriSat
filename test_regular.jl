using StyledStrings
using Printf

include("src/subdivision_regularity.jl")
using .SubdivisionRegularity

test_results = []
test_num = 1
test_names = []

if length(ARGS) > 0
    backend=ARGS[1]
else
    backend="cpu"
end


test_name = "Regularity of the \"Mother of all Examples\"}"
push!(test_names, test_name)
println("-")
println(styled"{bold, blue:Test $test_num, $test_name. Expect non-regular}")
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
    println(styled"{bold, red:failed}, falsely found that the Mother of all Examples is regular")
    push!(test_results, "failed: falsely found regular")
else
    println(styled"{bold, green:passed} The Mother of all Examples is not a regular subdivision.")
    push!(test_results, "")
end
test_num += 1


# Define triangulation directly as a vector of matrices of integer coordinates
triangulation_reg = [
    [0 0; 0 1; 1 1],
    [0 0; 1 1; 1 0]
    ]

elapsed_time = @elapsed begin
    regularity_result = is_regular(triangulation_reg)
end

if regularity_result
    println(styled"{bold, green:passed}.\n")
    push!(test_results, "")
else
    println(styled"{bold, red:failed}")
    push!(test_results, "failed")
end
test_num += 1

#!/usr/bin/env julia
#
# Generate the Hall simplices for all sequences s with s_i >= 2, s_1 >= s_d and
# sum(s) <= k, and write them to a file below <root>/Polytopes.
#
#     julia scripts/Hall_simplices.jl [k] [d] [outfile]
#
# The original wrote to the current working directory; the output path is now
# anchored to the repository. The generation logic is unchanged, only the
# indentation was straightened out (the nesting was misleading, not wrong).

include(joinpath(@__DIR__, "_setup.jl"))

# 1. Matrix construction

function generate_matrix(s::Vector{Int})
    d = length(s)
    M = zeros(Int, d + 1, d)
    for j in 1:d
        s_j = s[j]
        cutoff_row = d + 2 - j
        for i in cutoff_row:(d + 1)
            M[i, j] = s_j
        end
    end
    return M
end

# 2. Sequence generation

function find_sequences_le_k(k::Int, d_remaining::Int,
                             current_parts::Vector{Int},
                             results::Vector{Vector{Int}})
    current_sum = isempty(current_parts) ? 0 : sum(current_parts)

    if d_remaining == 1
        # Base case: picking the last element s_d, subject to
        #   s_d >= 2, s_d <= s_1, and sum <= k.
        s1 = current_parts[1]
        max_val = min(k - current_sum, s1)

        for s_i in 2:max_val
            push!(current_parts, s_i)
            push!(results, copy(current_parts))
            pop!(current_parts)
        end
        return
    end

    # Recursive step. Each remaining element must be at least 2, so
    #   max_s_i = k - current_sum - 2 * (d_remaining - 1).
    min_val = 2
    max_s_i = k - current_sum - (2 * (d_remaining - 1))

    for s_i in min_val:max_s_i
        push!(current_parts, s_i)
        find_sequences_le_k(k, d_remaining - 1, current_parts, results)
        pop!(current_parts)
    end
end

function get_all_sequences_le_k(k::Int, d::Int)
    results = Vector{Vector{Int}}()
    d >= 2 || error("d must be at least 2, got $d")   # base case reads s_1
    # The minimum sum is 2 * d because s_i >= 2.
    if k < 2 * d
        return results
    end
    find_sequences_le_k(k, d, Int[], results)
    return results
end

# 3. Formatting and file writing

function format_matrix(M::Matrix{Int})
    io = IOBuffer()
    R, C = size(M)
    for i in 1:R
        println(io, join([string(M[i, j]) for j in 1:C], " "))
    end
    return String(take!(io))
end

function write_matrices_to_file(k::Int, d::Int, filename::String)
    if k < 2 * d
        println("Error: k must be at least $(2 * d) for sequences where s_i >= 2.")
        return
    end

    sequences = get_all_sequences_le_k(k, d)
    count = 0

    mkpath(dirname(filename))
    open(filename, "w") do io
        for s in sequences
            M = generate_matrix(s)
            matrix_str = format_matrix(M)

            write(io, "# Matrix Dimension: ($(d+1)x$(d)), Sum(s): $(sum(s)), " *
                      "Max_k: $k, Seq: $(tuple(s...))\n")
            write(io, matrix_str)
            write(io, "\n")

            count += 1
        end
    end
    println("Successfully generated $count sequences " *
            "(s_i >= 2, s_1 >= s_d, sum <= $k) and wrote to '$filename'.")
end

# Defaults, overridable on the command line.
const K_VALUE = parse(Int, argordefault(1, "25"))   # total sum of the partitions
const D_VALUE = parse(Int, argordefault(2, "9"))    # dimension of the simplices
const OUTPUT_FILE = argordefault(
    3, polytope_path("hall_simplices_$(D_VALUE)Dim_sum_le_$(K_VALUE)"))

write_matrices_to_file(K_VALUE, D_VALUE, OUTPUT_FILE)

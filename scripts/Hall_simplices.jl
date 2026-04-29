# 1. Matrix Construction
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

# ----------------------------------------------------------------------
# 2. Optimized Sequence Generation
# ----------------------------------------------------------------------

function find_sequences_le_k(k::Int, d_remaining::Int, current_parts::Vector{Int}, results::Vector{Vector{Int}})
    current_sum = isempty(current_parts) ? 0 : sum(current_parts)

    if d_remaining == 1
        # Base Case: Picking the last element s_d
        # Constraint 1: s_d must be >= 2
        # Constraint 2: s_d must be <= s_1 (where s_1 is current_parts[1])
        # Constraint 3: sum must be <= k

        s1 = current_parts[1]
        max_val = min(k - current_sum, s1)

        for s_i in 2:max_val
            push!(current_parts, s_i)
            push!(results, copy(current_parts))
            pop!(current_parts)
        end
        return
    end

    # Recursive Step
    # Each remaining element must be at least 2.
    # So max_s_i = k - current_sum - (2 * (d_remaining - 1))
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
    # Minimum sum is now 2*d because s_i >= 2
    if k < 2*d
        return results
    end
    find_sequences_le_k(k, d, Int[], results)
    return results
end

# ----------------------------------------------------------------------
# 3. Formatting and File Writing (Updated Constraints)
# ----------------------------------------------------------------------

function format_matrix(M::Matrix{Int})
    io = IOBuffer()
    R, C = size(M)
    for i in 1:R
        row_str = join([string(M[i, j]) for j in 1:C], " ")
            println(io, row_str)
        end
        return String(take!(io))
    end

    function write_matrices_to_file(k::Int, d::Int, filename::String)
        if k < 2*d
            println("Error: k must be at least $(2*d) for sequences where s_i >= 2.")
                return
            end

            sequences = get_all_sequences_le_k(k, d)
            count = 0

            open(filename, "w") do io
                for s in sequences
                    M = generate_matrix(s)
                    matrix_str = format_matrix(M)

                    write(io, "# Matrix Dimension: ($(d+1)x$(d)), Sum(s): $(sum(s)), Max_k: $k, Seq: $(tuple(s...))\n")
                    write(io, matrix_str)
                    write(io, "\n")

                    count += 1
                end
            end
            println("Successfully generated $count sequences (s_i >= 2, s_1 >= s_d, sum <= $k) and wrote to '$filename'.")
        end

        # Example Usage
        const K_VALUE = 25  # The total sum of the partitions
        const D_VALUE = 9   # Dimension of the resulting simplices
        const OUTPUT_FILE = "hall_simplices_$(D_VALUE)Dim_sum<=$(K_VALUE)"

        write_matrices_to_file(K_VALUE, D_VALUE, OUTPUT_FILE)

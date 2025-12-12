# This script generates all (d+1)xd matrices corresponding to sequences
# (s1, ..., sd) of positive integers whose sum is less than or equal to k.

# ----------------------------------------------------------------------
# 1. Matrix Construction (Generalized)
# ----------------------------------------------------------------------

"""
    generate_matrix(s::Vector{Int})

Constructs the (d+1)xd matrix M from the d-element sequence s = (s1, ..., sd).
The matrix M is defined by:
M[i, j] = s[j] if i >= d + 2 - j, and 0 otherwise.
"""
function generate_matrix(s::Vector{Int})
    d = length(s)
    # Initialize the (d+1) x d matrix with zeros
    M = zeros(Int, d + 1, d)

    # j is the column index (1 to d), corresponding to s[j]
    for j in 1:d
        s_j = s[j]
        # The generalized cutoff row index: (d+1) - j + 1
        cutoff_row = d + 2 - j

        # Fill the matrix M[i, j] with s_j for i from cutoff_row to d+1
        for i in cutoff_row:(d + 1)
            M[i, j] = s_j
        end
    end
    return M
end

# ----------------------------------------------------------------------
# 2. Sequence Generation (Sum <= k, si >= 1)
# ----------------------------------------------------------------------

"""
    find_sequences_le_k(k::Int, d_remaining::Int, current_parts::Vector{Int}, results::Vector{Vector{Int}})

Recursive helper function to find all sequences (s1, ..., sd) where 
sum(s) <= k and all parts s_i >= 1.
d_remaining is the number of parts still to be chosen.
"""
function find_sequences_le_k(k::Int, d_remaining::Int, current_parts::Vector{Int}, results::Vector{Vector{Int}})
    
    current_sum = isempty(current_parts) ? 0 : sum(current_parts)

    if d_remaining == 1
        # Base case: The last part (s_d)
        # s_d can range from 1 up to k - current_sum, ensuring sum(s) <= k.
        max_val = k - current_sum
        
        for s_i in 1:max_val
            push!(current_parts, s_i)
            push!(results, copy(current_parts))
            pop!(current_parts)
        end
        return
    end

    # Recursive step: s_i can range from 1 up to the maximum value that
    # still leaves at least 1 for each of the remaining (d_remaining - 1) parts.
    
    # Maximum s_i = k - current_sum - (d_remaining - 1)
    # This ensures that current_sum + s_i + (d_remaining - 1) <= k
    max_s_i = k - current_sum - (d_remaining - 1)
    
    for s_i in 1:max_s_i
        push!(current_parts, s_i)
        find_sequences_le_k(k, d_remaining - 1, current_parts, results)
        pop!(current_parts) # Backtrack
    end
end

"""
    get_all_sequences_le_k(k::Int, d::Int)

Wrapper function to start the sequence generation.
"""
function get_all_sequences_le_k(k::Int, d::Int)
    results = Vector{Vector{Int}}()
    if k < d
        return results # Minimum sum of d positive parts is d
    end
    find_sequences_le_k(k, d, Int[], results)
    return results
end

# ----------------------------------------------------------------------
# 3. Matrix Formatting
# ----------------------------------------------------------------------

"""
    format_matrix(M::Matrix{Int})

Formats the matrix into a clean string representation where elements
in a row are separated by a space and each row is on a new line.
"""
function format_matrix(M::Matrix{Int})
    io = IOBuffer()
    R, C = size(M)
    for i in 1:R
        # Join elements of the row with a space
        row_str = join([string(M[i, j]) for j in 1:C], " ")
        println(io, row_str)
    end
    return String(take!(io))
end


# ----------------------------------------------------------------------
# 4. Main Logic for File Writing (Generalized)
# ----------------------------------------------------------------------

"""
    write_matrices_to_file(k::Int, d::Int, filename::String)

Finds all sequences (s1, ..., sd) where sum(s) <= k and s_i >= 1,
generates the corresponding matrix for each, and writes them to the file.
"""
function write_matrices_to_file(k::Int, d::Int, filename::String)
    if k < d
        println("Error: k must be greater than or equal to d for any sequence of positive parts (s_i >= 1) to exist.")
        return
    end

    sequences = get_all_sequences_le_k(k, d)
    count = 0

    open(filename, "w") do io
        for s in sequences
            # 1. Generate the matrix
            M = generate_matrix(s)

            # 2. Format the matrix
            matrix_str = format_matrix(M)
            
            # 3. Write to file, following user's new formatting instructions
            # Line 1: Commented dimension/sum info
            write(io, "# Matrix Dimension: ($(d+1)x$(d)), Sum(s): $(sum(s)), Max_k: $k\n")
            
            # Line 2 (Original composition line) is skipped.
            
            # Line 3: The matrix content
            write(io, matrix_str)
            
            # Line 4: Line space between separate matrices
            write(io, "\n") 

            count += 1
        end
    end
    println("Successfully generated $count sequences (sum <= $k, d=$d) and wrote $count matrices (representing vertices of the simplices) to '$filename'.")
end

# ----------------------------------------------------------------------
# 5. Example Usage (k=6, d=3)
# ----------------------------------------------------------------------

# Define the target sum k (maximum sum) and the number of parts d
const K_VALUE = 20 # The maximum sum of the sequence
const D_VALUE = 5 # The number of parts (d), resulting in a (d+1) x d matrix
const OUTPUT_FILE = "hall_simplices_$(D_VALUE)d.txt"

# Execute the main function
write_matrices_to_file(K_VALUE, D_VALUE, OUTPUT_FILE)

# Note: The output file will contain the results.
# Example output format (for s=(1, 1, 1) where k=6, d=3):
# # Matrix Dimension: (4x3), Sum(s): 3, Max_k: 6
# 0 0 0
# 0 0 1
# 0 1 1
# 1 1 1
#
# ... and so on for all sequences of 3 positive integers with sum <= 6.
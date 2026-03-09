module TrisatWrapper

export solve, itersolve

const LIB_EXT = Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so"
const LIB_PATH = joinpath(@__DIR__, "librust_sat" * LIB_EXT)

# =========================================================================
# Internal Helpers
# =========================================================================

# Flattens Vector{Vector{Int64}} into a 1D array separated by 0s.
# Also calculates the maximum variable index (num_vars).
function _prepare_cnf(cnf::Vector{Vector{Int64}})
    flat_cnf = Int64[]
    # Pre-allocate memory to make this ultra-fast
    sizehint!(flat_cnf, sum(length, cnf) + length(cnf))
    
    max_var = 0
    for clause in cnf
        for lit in clause
            push!(flat_cnf, lit)
            max_var = max(max_var, abs(lit))
        end
        push!(flat_cnf, 0) # 0 terminates the clause
    end
    
    return flat_cnf, max_var
end

function _call_rust_solver(cnf::Vector{Vector{Int64}}, find_all::Bool)::Int64
    if !isfile(LIB_PATH)
        error("Could not find compiled Rust library at: $LIB_PATH")
    end

    flat_array, num_vars = _prepare_cnf(cnf)
    
    # We pass the pointer, the length of the flat array, the max variable, and the flag
    result = ccall(
        (:solve_cnf_array, LIB_PATH), 
        Int64,                            # Return type
        (Ptr{Int64}, Csize_t, Csize_t, Bool), # Argument types
        pointer(flat_array),              # Ptr to the raw data
        length(flat_array),               # Length of the array
        num_vars,                         # Number of variables
        find_all                          # Mode flag
    )

    if result == -1
        error("Rust panic: Received empty or null array.")
    end

    return result
end

# =========================================================================
# Public API
# =========================================================================

function solve(cnf::Vector{Vector{Int64}})::Bool
    result = _call_rust_solver(cnf, false)
    return result == 1
end

function itersolve(cnf::Vector{Vector{Int64}})::Int64
    return _call_rust_solver(cnf, true)
end

end # module
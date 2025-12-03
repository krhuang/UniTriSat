module CadicalWrapper

using Libdl

const LIBCADICAL = joinpath(@__DIR__, "libcadical.so")

const STATUS_UNKNOWN = 0
const STATUS_SAT = 10
const STATUS_UNSAT = 20

mutable struct Solver
    ptr::Ptr{Cvoid}         
    is_solving::Bool        
    lock::ReentrantLock     

    function Solver()
        ptr = ccall((:ccadical_init, LIBCADICAL), Ptr{Cvoid}, ())
        solver = new(ptr, false, ReentrantLock())

        finalizer(release, solver)
        return solver
    end
end

"""
Release allocated memory
"""
function release(s::Solver)
    if s.ptr != C_NULL
        ccall((:ccadical_release, LIBCADICAL), Cvoid, (Ptr{Cvoid},), s.ptr)
        s.ptr = C_NULL
    end
end

"""
Add a single literal to build a clause
"""
function add_literal(s::Solver, lit::Int)
    ccall((:ccadical_add, LIBCADICAL), Cvoid, (Ptr{Cvoid}, Cint), s.ptr, Cint(lit))
end

"""
Helper: Adds a full clause at once
"""
function add_clause(s::Solver, clause::Vector{Int})
    lock(s.lock) do
        if s.is_solving
            error("Unable to add clause, pause the solver first by calling 'interrupt(solver)'")
        end
        for lit in clause
            add_literal(s, lit)
        end
        add_literal(s, 0) # Terminate clause
    end
end

"""
Pauses the solver
"""
function interrupt(s::Solver)
    ccall((:ccadical_terminate, LIBCADICAL), Cvoid, (Ptr{Cvoid},), s.ptr)
end

"""
Starts the solver in the current thread (blocking)
"""
function solve_blocking(s::Solver)
    lock(s.lock) do
        s.is_solving = true
    end
    res = try
        ccall((:ccadical_solve, LIBCADICAL), Cint, (Ptr{Cvoid},), s.ptr)
    finally
        lock(s.lock) do
            s.is_solving = false
        end
    end

    return res
end

"""
Starts solver in the background and returns a julia task
"""
function solve_async(s::Solver)
    if s.is_solving
        error("Solver already running!")
    end

    task = Threads.@spawn begin
        res = solve_blocking(s)
#         if res == STATUS_SAT
#             println("\n[Async] Solver finished: SAT")
#         elseif res == STATUS_UNSAT
#             println("\n[Async] Solver finished: UNSAT")
#         else
#             println("\n[Async] Solver finished: UNKNOWN (Interrupted)")
#         end
        return res
    end

    return task
end

"""
Check the value of a literal to extract the actual assignment
"""
function val(s::Solver, lit::Int)
    return ccall((:ccadical_val, LIBCADICAL), Cint, (Ptr{Cvoid}, Cint), s.ptr, Cint(lit))
end

end # module
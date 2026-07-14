module Solving

# Number of buffered redundant intersection clauses that justifies pausing the
# SAT solver to inject them (incremental solving only). Tunable: larger values
# interrupt the solver less often, smaller values get information to the
# solver sooner.
const SOLVER_UPDATE_THRESHOLD = 500_000

using PicoSAT

# Include the Cadical wrapper
include("CadicalWrapper.jl")
using .CadicalWrapper

include("subdivision_regularity.jl")
using .SubdivisionRegularity

using ..Structs
using ..Helpers
using ..BasicComputations
using Base.Threads

export solve_picosat, solve_cadical_incremental, solve_cadical_standard, solve_parallel

# ============================================================================
# Shared solution bookkeeping
# ============================================================================
#
# Every solver enumerates raw SAT solutions (unimodular triangulations) and
# must classify them according to the configuration (regular / flag /
# quadratic), update counters, remember solutions, and decide when to stop.
# This logic used to be copy-pasted into each solver and had drifted apart;
# it now lives in exactly one place.

mutable struct SolveState
    solution_simplices::Vector{Vector{Matrix{Int}}}
    first_solution_simplices::Vector{Matrix{Int}}
    n_found::Int      # all triangulations seen
    n_regular::Int    # regular ones (only tested when config.regular)
    n_flag::Int       # flag ones (only tested when config.flag_triangulation;
                      #   when config.regular is also set, flagness is only
                      #   tested for regular triangulations, as before)
    n_quadratic::Int  # regular and flag (only when both filters are active)
end

SolveState() = SolveState(Vector{Vector{Matrix{Int}}}(), Vector{Matrix{Int}}(), 0, 0, 0, 0)

as_result_tuple(st::SolveState) = (
    st.solution_simplices,
    st.first_solution_simplices,
    st.n_found,
    st.n_regular,
    st.n_flag,
    st.n_quadratic,
)

# Converts a list of true variable indices into the corresponding triangulation
extract_simplices(P::Matrix{Int}, S_indices, sol_indices) =
    [convert(Matrix{Int}, P[collect(S_indices[i]), :]) for i in sol_indices]

"""
    record_solution!(st, simplices, config, show_running_updates) -> Bool

Classify one triangulation, update the counters and stored solutions in `st`,
and return `true` iff the search should terminate, i.e. a triangulation of the
requested type was found and `config.find_all == false`.
"""
function record_solution!(st::SolveState, simplices::Vector{Matrix{Int}},
                          config::Config, show_running_updates::Bool)
    st.n_found += 1
    if isempty(st.first_solution_simplices)
        st.first_solution_simplices = simplices
    end

    # Does this triangulation satisfy all requested filters?
    is_target = true

    if config.regular
        if is_regular(simplices)
            st.n_regular += 1
        else
            is_target = false
        end
    end

    # When both filters are active we are hunting quadratic triangulations;
    # flagness is then only tested for regular triangulations (as before).
    if is_target && config.flag_triangulation
        if is_flag_triangulation(simplices)
            st.n_flag += 1
            if config.regular
                st.n_quadratic += 1
            end
        else
            is_target = false
        end
    end

    if is_target
        if config.return_triangulations == "all" ||
           (config.return_triangulations == "first" && isempty(st.solution_simplices))
            push!(st.solution_simplices, simplices)
        end
        if !config.find_all
            return true
        end
    elseif show_running_updates
        ghost_print(" ($(st.n_found) triangulations checked, still searching for the requested type)")
    end

    return false
end

# ============================================================================
# PicoSAT
# ============================================================================

function solve_picosat(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config,
                       show_running_updates::Bool, stop_signal::Threads.Atomic{Bool})
    st = SolveState()

    for solution in PicoSAT.itersolve(cnf)
        # Check if another thread has already signaled to stop
        if stop_signal[]
            break
        end

        if show_running_updates && st.n_found > 0 && st.n_found % 1000 == 0
            ghost_print(" ($(st.n_found) triangulations found)")
        end

        sol_indices = findall(l -> l > 0, solution)
        simplices = extract_simplices(P, S_indices, sol_indices)

        if record_solution!(st, simplices, config, show_running_updates)
            Threads.atomic_cas!(stop_signal, false, true)
            break
        end
    end

    return as_result_tuple(st)
end

# ============================================================================
# CaDiCaL, standard (non-incremental) enumeration
# ============================================================================

function solve_cadical_standard(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config,
                                show_running_updates::Bool, stop_signal::Threads.Atomic{Bool})
    st = SolveState()
    num_simplices = length(S_indices)

    solver = CadicalWrapper.Solver()
    try
        for clause in cnf
            CadicalWrapper.add_clause(solver, clause)
        end

        while !stop_signal[]
            if show_running_updates && st.n_found > 0 && st.n_found % 1000 == 0
                ghost_print(" ($(st.n_found) triangulations found)")
            end

            res = CadicalWrapper.solve_blocking(solver)
            if res != CadicalWrapper.STATUS_SAT
                break # UNSAT (or unknown): enumeration finished
            end

            sol_indices = [i for i in 1:num_simplices if CadicalWrapper.val(solver, i) > 0]
            simplices = extract_simplices(P, S_indices, sol_indices)

            if record_solution!(st, simplices, config, show_running_updates)
                Threads.atomic_cas!(stop_signal, false, true)
                break
            end

            # Block this assignment and search for the next triangulation
            CadicalWrapper.add_clause(solver, -sol_indices)
        end
    finally
        CadicalWrapper.release(solver)
    end

    return as_result_tuple(st)
end

# ============================================================================
# CaDiCaL, incremental solving
# ============================================================================

function solve_cadical_incremental(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, dim::Int,
                                   config::Config, show_running_updates::Bool, log_verbose::Function)
    st = SolveState()
    num_simplices = length(S_indices)

    if nthreads() < 3
        @warn("Incremental solving works best with at least 3 Julia threads " *
              "(solver, clause generation, coordination); running with $(nthreads()). " *
              "Consider starting Julia with `julia -t auto`.")
    end

    solver = CadicalWrapper.Solver()
    for clause in cnf
        CadicalWrapper.add_clause(solver, clause)
    end

    log_verbose("      Incremental mode: solving the reduced formula while streaming redundant intersection clauses...")

    # ---- background clause generation --------------------------------------
    cpu_simplices = CPUIntersection.prepare_simplices_cpu(P, S_indices, Val(dim))

    # Workers push one batch per "row" i: the conflict clauses of simplex i
    # against all simplices j > i. Batching keeps channel contention
    # negligible even when millions of clauses are produced.
    clause_channel = Channel{Vector{Vector{Int}}}(max(64, 4 * nthreads()))
    next_row = Threads.Atomic{Int}(1)
    search_finished = Threads.Atomic{Bool}(false)
    generator_error = Threads.Atomic{Bool}(false)

    num_workers = max(1, nthreads() - 2) # leave room for the solver and the coordinator
    generator_tasks = map(1:num_workers) do worker_id
        Threads.@spawn begin
            try
                while !search_finished[]
                    i = Threads.atomic_add!(next_row, 1)
                    if i >= num_simplices
                        break
                    end
                    batch = CPUIntersection.check_intersections_range_cpu(cpu_simplices, i, i + 1, num_simplices)
                    if !isempty(batch)
                        put!(clause_channel, batch)
                    end
                end
            catch e
                if e isa InvalidStateException
                    # The channel was closed because the search finished early: benign.
                else
                    @error "Intersection generator $worker_id failed" exception = (e, catch_backtrace())
                    generator_error[] = true
                end
            end
        end
    end

    # Close the channel once every worker is done, so the coordinator can
    # distinguish "no clauses right now" from "no clauses ever again".
    closer_task = Threads.@spawn begin
        try
            foreach(t -> (try; wait(t); catch; end), generator_tasks)
        finally
            close(clause_channel)
        end
    end

    # ---- coordinator --------------------------------------------------------
    buffer = Vector{Vector{Int}}()
    total_injected = 0
    solve_task = CadicalWrapper.solve_async(solver)

    # PRECONDITION for calling this: the solver is idle (solve_task finished).
    flush_buffer! = function ()
        for clause in buffer
            CadicalWrapper.add_clause(solver, clause)
        end
        total_injected += length(buffer)
        empty!(buffer)
    end

    try
        while true
            if generator_error[]
                @error "Aborting incremental solving: an intersection generator failed."
                break
            end

            # 1) Harvest whatever the generators have produced so far.
            while isready(clause_channel)
                append!(buffer, take!(clause_channel))
            end
            generation_done = !isopen(clause_channel) && !isready(clause_channel)

            # 2) A finished solver run is always handled *before* anything is
            #    added, so a SAT/UNSAT result that raced with an interrupt can
            #    never be lost.
            if istaskdone(solve_task)
                res = fetch(solve_task)

                if res == CadicalWrapper.STATUS_SAT
                    sol_indices = [i for i in 1:num_simplices if CadicalWrapper.val(solver, i) > 0]
                    simplices = extract_simplices(P, S_indices, sol_indices)

                    if record_solution!(st, simplices, config, show_running_updates)
                        break
                    end
                    if show_running_updates && st.n_found % 1000 == 0
                        ghost_print(" ($(st.n_found) triangulations found)")
                    end

                    # Keep enumerating: block this assignment. The solver is
                    # conveniently idle, so flush pending clauses too.
                    CadicalWrapper.add_clause(solver, -sol_indices)
                    flush_buffer!()
                    solve_task = CadicalWrapper.solve_async(solver)

                elseif res == CadicalWrapper.STATUS_UNSAT
                    # All clauses ever added are sound, so this is definitive
                    # even if clause generation is still running.
                    break

                else # STATUS_UNKNOWN: interrupted (or spurious termination)
                    had_work = !isempty(buffer)
                    flush_buffer!()
                    if show_running_updates && had_work
                        ghost_print("   (incremental) intersection clauses injected: $total_injected")
                    end
                    if !had_work
                        # Nothing was waiting; avoid a hot interrupt/resume loop.
                        sleep(0.001)
                    end
                    solve_task = CadicalWrapper.solve_async(solver)
                end
                continue
            end

            # 3) Solver still running: pause it when enough clauses have piled
            #    up, or when generation has finished and the leftovers should
            #    go in. The result of the interrupted run is handled at the
            #    top of the loop.
            if length(buffer) >= SOLVER_UPDATE_THRESHOLD || (generation_done && !isempty(buffer))
                CadicalWrapper.interrupt(solver)
                wait(solve_task)
                continue
            end

            # 4) Nothing to coordinate: either wait for the solver (generation
            #    is over and everything has been injected) or nap briefly
            #    while the generators work.
            if generation_done
                wait(solve_task)
            else
                sleep(0.001)
            end
        end
    finally
        # Tear everything down on every exit path.
        search_finished[] = true
        close(clause_channel) # unblocks workers stuck in put!
        if !istaskdone(solve_task)
            CadicalWrapper.interrupt(solver)
            try
                wait(solve_task)
            catch
            end
        end
        foreach(t -> (try; wait(t); catch; end), generator_tasks)
        try
            wait(closer_task)
        catch
        end
        CadicalWrapper.release(solver)
    end

    log_verbose("      Injected $total_injected redundant intersection clauses during solving.")

    return as_result_tuple(st)
end

# ============================================================================
# Parallel solving
# ============================================================================

function solve_parallel(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, config::Config, show_running_updates::Bool)
    num_threads = Threads.nthreads()

    # 1. find generic point and central simplices
    generic_point = find_generic_point(P, internal_faces(P, size(P, 2)), Val(size(P, 2)))
    central_indices_map = compute_central_indices(P, S_indices, generic_point)

    solve_function = nothing
    if config.solver == "cadical"
        solve_function = solve_cadical_standard
    elseif config.solver == "picosat"
        solve_function = solve_picosat
    else
        error("Unsupported solver for parallel execution: $(config.solver)")
    end

    # 2. split central simplices into groups for each thread (round-robin)
    central_groups = [Int[] for _ in 1:num_threads]
    for (i, idx) in enumerate(central_indices_map)
        push!(central_groups[mod1(i, num_threads)], idx)
    end

    # 3. prepare thread-local storage for results
    solution_simplices_threads = [Vector{Vector{Matrix{Int}}}() for _ in 1:num_threads]
    first_solution_threads     = [Vector{Matrix{Int}}() for _ in 1:num_threads]

    # Thread-isolated quantitative tracking
    num_triangulations_threads = zeros(Int, num_threads)
    num_regular_threads        = zeros(Int, num_threads)
    num_flag_threads           = zeros(Int, num_threads)
    num_quadratic_threads      = zeros(Int, num_threads)

    # Shared stop signal for early termination across all threads
    found_solution = Threads.Atomic{Bool}(false)

    # 4. thread i solves cnf + [central_group[i]...] using the standard solver.
    # This limits thread i to solutions where a simplex from its central group
    # is used; since every triangulation contains exactly one central simplex,
    # the groups partition the solution space.
    Threads.@threads for tid in 1:num_threads
        group = central_groups[tid]
        if isempty(group)
            continue
        end

        # Create a shallow copy of the CNF and add this thread's group constraint
        local_cnf = copy(cnf)
        push!(local_cnf, group)

        # Call the standard solving logic, suppressing live updates to avoid thread clutter
        sol_simp, first_sol, num_found, num_reg, num_flag, num_quad = solve_function(
            local_cnf, P, S_indices, config, false, found_solution
        )

        # Store the returned values into the isolated thread arrays
        solution_simplices_threads[tid] = sol_simp
        first_solution_threads[tid]     = first_sol

        num_triangulations_threads[tid] = num_found
        num_regular_threads[tid]        = num_reg
        num_flag_threads[tid]           = num_flag
        num_quadratic_threads[tid]      = num_quad
    end

    # 5. aggregate the results from all threads
    solution_simplices       = Vector{Vector{Matrix{Int}}}()
    first_solution_simplices = Vector{Matrix{Int}}()

    # Compute totals
    number_of_triangulations_found           = sum(num_triangulations_threads)
    number_of_regular_triangulations_found   = sum(num_regular_threads)
    number_of_flag_triangulations_found      = sum(num_flag_threads)
    number_of_quadratic_triangulations_found = sum(num_quadratic_threads)

    for tid in 1:num_threads
        append!(solution_simplices, solution_simplices_threads[tid])

        # Capture the very first valid solution found across threads
        if isempty(first_solution_simplices) && !isempty(first_solution_threads[tid])
            first_solution_simplices = first_solution_threads[tid]
        end
    end

    # If only the first triangulation is requested, truncate the combined list
    if config.return_triangulations == "first" && length(solution_simplices) > 1
        resize!(solution_simplices, 1)
    end

    # Print a final summary if requested
    if show_running_updates
        ghost_print("Parallel search finished: $number_of_triangulations_found triangulations found " *
                    "($number_of_regular_triangulations_found regular, " *
                    "$number_of_flag_triangulations_found flag, " *
                    "$number_of_quadratic_triangulations_found quadratic).")
    end

    return (
        solution_simplices,
        first_solution_simplices,
        number_of_triangulations_found,
        number_of_regular_triangulations_found,
        number_of_flag_triangulations_found,
        number_of_quadratic_triangulations_found
    )
end

end

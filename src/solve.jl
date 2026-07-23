module Solving

# ---- Tunables for incremental solving ------------------------------------
# Number of buffered redundant intersection clauses that justifies pausing the
# SAT solver to inject them. Larger values interrupt the solver less often,
# smaller values get information to the solver sooner.
const SOLVER_UPDATE_THRESHOLD = 500_000
# Backpressure: the coordinator stops draining the generator channel once this
# many clauses are waiting for injection. The (bounded) channel then fills up
# and the generators block on it, so clause production is throttled to the
# rate at which the solver can actually absorb clauses. Without this cap the
# buffer grows without bound whenever generation outpaces injection.
const BUFFER_HARD_CAP = 2 * SOLVER_UPDATE_THRESHOLD
# Generators split their output into chunks of at most this many clauses per
# channel message, so that a single dense "row" (one simplex against all
# later ones) cannot blow up the channel's memory footprint.
const GENERATOR_CHUNK_SIZE = 100_000
# Global budget for injected redundant clauses. On instances where most
# simplex pairs intersect (e.g. many simplices crammed into a small volume,
# like the 0/1 cubes) the full pairwise clause set can reach 10^10 clauses --
# impossible to generate or store. Since the reduced formula is already
# equivalent, it is sound to stop strengthening at ANY point; past this budget
# the generators are shut down and the solver runs with what it has. Tune to
# available RAM (binary clauses cost some tens of bytes inside CaDiCaL).
const MAX_INJECTED_CLAUSES = 20_000_000
# Minimum uninterrupted time the solver gets between clause injections; the
# slice grows on every interrupt. Early (high-value) clauses still arrive
# quickly, while later injections leave the solver alone for longer and
# longer. Without this gate, fast generators refill the buffer every few
# milliseconds, the solver is interrupted before it can make any progress,
# and the whole run degenerates into pure clause shoveling.
const SOLVE_SLICE_INITIAL_SECONDS = 0.5
const SOLVE_SLICE_GROWTH = 1.5
const SOLVE_SLICE_MAX_SECONDS = 10.0

using PicoSAT
using Printf

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
        CadicalWrapper.add_clauses(solver, cnf)

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
#
# The CNF handed to this function is the *reduced equivalent formula* built in
# compute_intersections_incremental: non-emptiness, face-covering clauses, an
# exactly-one structure over the "central" simplices (those containing a fixed
# generic point), and the hyperplane-separation clauses. This formula already
# has exactly the unimodular triangulations of P as its solutions, but it
# gives the solver little to propagate on.
#
# Therefore, while the coordinator runs CaDiCaL on that formula, worker tasks
# compute pairwise intersection clauses in the background. These clauses are
# logically redundant (every triangulation satisfies them) but sharpen unit
# propagation. Whenever enough of them have been buffered, the solver is
# interrupted, the clauses are injected, and the solver resumes; CaDiCaL
# keeps its learned clauses across such restarts.
#
# Flow control (all three are essential; see the constants at the top):
#   * The channel is bounded and the coordinator stops draining it while its
#     own backlog exceeds BUFFER_HARD_CAP, so the generators block instead of
#     flooding memory.
#   * A single row is split into GENERATOR_CHUNK_SIZE chunks.
#   * At most MAX_INJECTED_CLAUSES are ever injected; past that, generation
#     is shut down and the solver runs on what it has. This is sound because
#     the reduced formula is already equivalent -- the streamed clauses only
#     help propagation.
#
# Soundness:
#   * Every injected clause (redundant intersection clause or blocking clause
#     of an already-reported solution) is satisfied by every not-yet-reported
#     triangulation. Hence a SAT answer is always a genuine triangulation and
#     an UNSAT answer is always definitive -- even while clause generation is
#     still running or after it has been cut short.
#   * Only the coordinator ever touches the solver (add / solve / val);
#     workers communicate exclusively through the channel. CaDiCaL's
#     terminate call is the one documented asynchronous exception.

format_count(n::Integer) =
    n >= 1_000_000 ? @sprintf("%.1fM", n / 1_000_000) :
    n >= 10_000    ? @sprintf("%.1fk", n / 1_000)     : string(n)

function solve_cadical_incremental(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, dim::Int,
                                   config::Config, show_running_updates::Bool, log_verbose::Function)
    st = SolveState()
    num_simplices = length(S_indices)
    num_rows = max(num_simplices - 1, 0)

    if nthreads() < 3
        @warn("Incremental solving works best with at least 3 Julia threads " *
              "(solver, clause generation, coordination); running with $(nthreads()). " *
              "Consider starting Julia with `julia -t auto`.")
    end

    solver = CadicalWrapper.Solver()
    CadicalWrapper.add_clauses(solver, cnf)

    log_verbose("      Incremental mode: solving the reduced formula while streaming redundant intersection clauses...")

    # ---- background clause generation --------------------------------------
    cpu_simplices = CPUIntersection.prepare_simplices_cpu(P, S_indices, Val(dim))

    clause_channel = Channel{Vector{Vector{Int}}}(max(16, 2 * nthreads()))
    next_row = Threads.Atomic{Int}(1)
    rows_completed = Threads.Atomic{Int}(0)
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
                    # Push in bounded chunks: a single dense row can hold
                    # hundreds of thousands of clauses.
                    for chunk_start in 1:GENERATOR_CHUNK_SIZE:length(batch)
                        if search_finished[]
                            break
                        end
                        chunk_end = min(chunk_start + GENERATOR_CHUNK_SIZE - 1, length(batch))
                        put!(clause_channel, batch[chunk_start:chunk_end])
                    end
                    Threads.atomic_add!(rows_completed, 1)
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
    clause_budget_exhausted = false
    last_status_time = 0.0
    solve_slice = SOLVE_SLICE_INITIAL_SECONDS
    solve_task = CadicalWrapper.solve_async(solver)
    last_resume_time = time()

    # PRECONDITION for calling this: the solver is idle (solve_task finished).
    flush_buffer! = function ()
        if !isempty(buffer)
            CadicalWrapper.add_clauses(solver, buffer)
            total_injected += length(buffer)
            empty!(buffer)
        end
    end

    # Live feedback line (throttled). This is the line the user watches during
    # long incremental runs, so it always carries the injected-clause count.
    print_status! = function (force::Bool = false)
        if !show_running_updates
            return
        end
        now_t = time()
        if force || now_t - last_status_time > 0.5
            last_status_time = now_t
            if clause_budget_exhausted
                ghost_print(" incremental: clause budget reached ($(format_count(total_injected)) clauses added), solver running | $(st.n_found) triangulations found")
            else
                ghost_print(" incremental: $(format_count(total_injected)) intersection clauses added, pair rows $(min(rows_completed[], num_rows))/$(num_rows) | $(st.n_found) triangulations found")
            end
        end
    end

    try
        while true
            if generator_error[]
                @error "Aborting incremental solving: an intersection generator failed."
                break
            end

            # 0) Enforce the global clause budget. Sound to stop at any point:
            #    the reduced formula is already equivalent.
            if !clause_budget_exhausted && total_injected >= MAX_INJECTED_CLAUSES
                clause_budget_exhausted = true
                search_finished[] = true
                close(clause_channel)
                while isready(clause_channel)
                    take!(clause_channel) # discard: injection is over
                end
                empty!(buffer)
                log_verbose("      Clause budget of $(MAX_INJECTED_CLAUSES) injected clauses reached; continuing with the solver only.")
                print_status!(true)
            end

            # 1) Harvest, with backpressure: while the backlog is large, the
            #    channel is left alone, it fills up, and the generators block.
            if !clause_budget_exhausted
                while length(buffer) < BUFFER_HARD_CAP && isready(clause_channel)
                    append!(buffer, take!(clause_channel))
                end
            end
            generation_done = !isopen(clause_channel) && !isready(clause_channel)

            print_status!()

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
                    print_status!(true)

                    # Keep enumerating: block this assignment. The solver is
                    # conveniently idle, so flush pending clauses too.
                    CadicalWrapper.add_clause(solver, -sol_indices)
                    flush_buffer!()
                    solve_task = CadicalWrapper.solve_async(solver)
                    last_resume_time = time()

                elseif res == CadicalWrapper.STATUS_UNSAT
                    # All clauses ever added are sound, so this is definitive
                    # even if clause generation is unfinished or was cut short.
                    break

                else # STATUS_UNKNOWN: interrupted (or spurious termination)
                    had_work = !isempty(buffer)
                    flush_buffer!()
                    if had_work
                        print_status!(true)
                    else
                        # Nothing was waiting; avoid a hot interrupt/resume loop.
                        sleep(0.001)
                    end
                    solve_task = CadicalWrapper.solve_async(solver)
                    last_resume_time = time()
                end
                continue
            end

            # 3) Solver still running: pause it when enough clauses have
            #    piled up, or when generation has finished and the leftovers
            #    should go in -- but never before the solver had its minimum
            #    uninterrupted slice. While the slice runs out, the capped
            #    buffer keeps the generators blocked, throttling generation to
            #    the rate the solver can actually absorb. The result of the
            #    interrupted run is handled at the top of the loop.
            ready_to_inject = length(buffer) >= SOLVER_UPDATE_THRESHOLD ||
                              (generation_done && !isempty(buffer))
            if ready_to_inject && time() - last_resume_time >= solve_slice
                solve_slice = min(solve_slice * SOLVE_SLICE_GROWTH, SOLVE_SLICE_MAX_SECONDS)
                CadicalWrapper.interrupt(solver)
                wait(solve_task)
                continue
            end

            # 4) Nothing to coordinate right now: either everything available
            #    is injected and generation is over (just wait for the solver,
            #    waking up briefly to keep the status line fresh), or we are
            #    waiting for generators / for the solve slice to elapse.
            if generation_done && isempty(buffer)
                if show_running_updates
                    sleep(0.25)
                    print_status!()
                else
                    wait(solve_task)
                end
            else
                sleep(0.001)
            end
        end
    finally
        # Tear everything down on every exit path.
        search_finished[] = true
        close(clause_channel) # unblocks workers stuck in put!
        while isready(clause_channel)
            take!(clause_channel) # free channel memory promptly
        end
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

function solve_parallel(cnf::Vector{Vector{Int}}, P::Matrix{Int}, S_indices, internal_faces_set,
                        config::Config, show_running_updates::Bool)
    num_threads = Threads.nthreads()

    # 1. find generic point and central simplices. The internal faces were
    # already computed in Step 3 of process_polytope and are passed in;
    # recomputing them here used to build a further CDD-exact polyhedron and
    # re-enumerate all C(n,d) subsets for every single polytope.
    generic_point = find_generic_point(P, internal_faces_set, Val(size(P, 2)))
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

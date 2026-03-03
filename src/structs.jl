module Structs

export StepStat, StatAggregator, Config, TriangulationResult, RunResult

# a struct to keep track of the timings of the separate operations
struct StepStat
    name::String
    duration_s::Float64
    alloc_bytes::Int64
end

# a struct to aggregate statistics on the fly, avoiding the storage of every single step result
mutable struct StatAggregator
    total_time::Float64
    max_time::Float64
    total_alloc::Int64
    max_alloc::Int64
    count::Int
end
# Initializer for the aggregator
StatAggregator() = StatAggregator(0.0, 0.0, 0, 0, 0)

mutable struct Config
    terminal_output::String
    unimodular::Bool
    intersection_backend::String
    regular::Bool
    find_all::Bool
    validate::Bool
    plot::Bool
    use_normaliz::Bool
    return_triangulations::String
    solver::String
    incremental_solving::Bool
    check_full_dimensionality::Bool
    enable_parallel::Bool
end

mutable struct TriangulationResult
    solution_simplices::Vector{Vector{Matrix{Int}}}
    number_of_triangulations_found::Int
    number_of_regular_triangulations_found::Int
    minimal_log::String
    total_time::Float64
    step_stats::Vector{StepStat}
end

mutable struct RunResult
    triangulation_results::Vector{TriangulationResult}
    number_triangulatable::Int
    number_regularly_triangulatable::Int
    total_number_of_triangulations_found::Int
    total_number_of_regular_triangulations_found::Int
    total_time::Float64
end

end
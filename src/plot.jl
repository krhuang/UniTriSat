module Plot

export plot

    function format_simplices_for_plotter(simplices::Vector{Matrix{Int}})
        return "[" * join(["[" * join(["[$(join(v, ","))]" for v in eachrow(s)], ",") * "]" for s in simplices], ",") * "]"
    end

    # Plots a 2- or 3-dimensional triangulation via src/plot_triangulation.py (requires Python with plotly and numpy).
    function plot(initial_vertices::Matrix{Int}, dim::Int, first_solution_simplices::Vector{Matrix{Int}})
        if !(dim in (2, 3))
            @warn("Plotting is only supported for 2- and 3-dimensional polytopes; skipping plot for dimension $dim.")
            return nothing
        end
        temp_path, temp_io = mktemp()
        try
            write(temp_io, format_simplices_for_plotter(first_solution_simplices))
            close(temp_io)
            script = joinpath(@__DIR__, "plot_triangulation.py")
            run(`python $script $temp_path`)
        finally
            rm(temp_path, force=true)
        end
        return nothing
    end

end

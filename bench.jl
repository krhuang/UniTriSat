using StyledStrings
using Printf

include("Triangulate.jl")
using .Triangulate
include("src/helpers.jl")
using .Helpers


terminal_output = "running, table, final" #initial, running, table, final

bench_results = []
bench_num = 1

bench_names = [
    "Vol 18 3D",
    "Big 3D",
    "Vol 16 4D",
    "Vol 16 5D",
    "Vol 13 6D"
    ]
paths = [
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v18.txt",
    "Polytopes/Big3D",
    "Polytopes/small-lattice-polytopes/data/4-polytopes/v16.txt",
    "Polytopes/small-lattice-polytopes/data/5-polytopes/v16.txt",
    "Polytopes/small-lattice-polytopes/data/6-polytopes/v13.txt",
    ]
backends = ["gpu", "cpu"]

n = length(bench_names)

for backend in backends
    for (i, name) in enumerate(bench_names)
        println("-")
        println(styled"{bold, blue:Benchmark $i/$n: $name. Running on $backend}")
        println("-")
        time  = @timed triangulate(
            paths[i],
            terminal_output=terminal_output, intersection_backend=backend
            )
        println("Finished in $(format_duration(time.time))")
        push!(bench_results, time.time)
    end
end

for i in 1:length(n)
    for (j,backend) in enumerate(backends)
        time = format_duration(bench_results[(j-1)*n+i])
        println("Benchmark $i: $(bench_names[i]), ran on $backend: $time")
    end
end

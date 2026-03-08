Finding unimodular triangulations and regular unimodular triangulations of lattice polytopes, via conversion to a SAT equation. `UniTriSat` can handle larger examples than were previously possible--it finds a unimodular triangulation of a 3-polytope with 50 lattice points in about 2 seconds. 

**UniTriSat is still under active development. An upcoming future version will be substantially faster, especially for finding regular unimodular triangulations. If it fails for a particularly large example of yours, please email us at krhuang5@gmail.com**

Quick Start Guide
=================

**1)** Install Julia (e.g. via juliaup)

**2)** Add the package via Julia's package manager: 

`import Pkg; Pkg.add(url="https://github.com/krhuang/UniTriSat/")`

**3)** Run on your favorite lattice polytope

```
using UniTriSat;
P = [ 1 0  0 0 0; -1 0 0 0 0; 0 1 0  0 0; 0 -1 0 0 0; 0 0 1 0  0;
      0 0 -1 0 0;  0 0 0 1 0; 0 0 0 -1 0; 0  0 0 0 1; 0 0 0 0 -1]

triangulate(P, terminal_output="running,table,final")`
```

You can also add Balletti's database of lattice polytopes to the directory `Polytopes`, e.g.

`git clone https://github.com/gabrieleballetti/small-lattice-polytopes /Polytopes`

Once this is done, you can run the code via the `triangulate` entry point function like so:

```
using UniTriSat

triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt")
```

See the function `triangulate` in `src/UniTriSat.jl` to see what options the function takes.

Options and return types
========
The triangulate function takes the following arguments:
```
polytopes; 
intersection_backend: "cpu" or "gpu", default is "cpu", 
unimodular: flag to toggle wether to restrict to unimodular simplices, default true, 
regular: flag to toggle wether to restrict to regular triangulations, defualt false,, 
find_all: flag to toggle wether to find all valid triangulations, default false, 
log_file: a path to a log file, e.g. "logs/my_run.log", leave empty for no logging 
terminal_output: string controlling what is printed to the terminal, any subset of "initial,running,table,final" is accepted
      initial: prints an initial summary of the run parameters
      running: prints intermediate results about the polytopes which are already done
      table: prints a table summarizing run times and memory usage split by the different operations (like SAT solving, generating intersection clauses ets.)
      final: prints a final summary of the resuls 
validate: NOT YET IMPLEMENTED, flag to toggle wether the triangulations found should be checked by some other algorithm, 
plot: flag to toggle wether the triangulations found should be printed. If find_all is true, then the first found triangulation is plotted.
      In higher dimensions, the projection of the triangulation to the 3-faces of the polytopes are plotted
use_normaliz: flag to toggle wether to use Normaliz to compute interior lattice points. Normaliz is faster, but we encountered stability issues, 
return_triangulations: string to controll what to return
      "all": return all found triangulations
      "first": return the first triangulations
      "": dont return triangulations
```

The results are returned as a RunResult struct which countains many TriangulationsResult structs, as follows:
```
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
```

Tests
=============

You can run a tests set by doing `julia test.jl`.

We are in need of non-regular unimodular triangulations for our unit testing. If you know of any, other than the "mother of all examples" please write us. 


UniTriSat: Unimodular Triangulations via SATISFIABILITY
=================

A novel algorithm for finding (regular, flag) unimodular triangulations of lattice polytopes, via conversion to a SAT equation. `UniTriSat` can handle larger examples than were previously possible with [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/index.html) or [mptopcom](https://polymake.org/doku.php/mptopcom)--it finds a unimodular triangulation of a 3-polytope with 50 lattice points in about 30 seconds. 

This is project of [Kyle Huang](https://krhuang.github.io/), [Robert Lauff](https://page.math.tu-berlin.de/~lauff/), and Charles Zhang. 

**UniTriSat is still under active development. If it fails for a particular example of yours, please email us at krhuang5@gmail.com**

Quick Start Guide
=================

**1)** Install Julia (e.g. via [juliaup](https://github.com/JuliaLang/juliaup))

**2)** Add the package via Julia's package manager: 

`import Pkg; Pkg.add(url="https://github.com/krhuang/UniTriSat/")`

**3)** Run on your favorite lattice polytope

```
using UniTriSat;
P = [ 1 0  0 0 0; -1 0 0 0 0; 0 1 0  0 0; 0 -1 0 0 0; 0 0 1 0  0;
      0 0 -1 0 0;  0 0 0 1 0; 0 0 0 -1 0; 0  0 0 0 1; 0 0 0 0 -1]

triangulate(P, terminal_output="running,table,final")`
```

The code can also run on plain text files of lattice polytopes, e.g. Balletti's database, like so:

```
using UniTriSat

triangulate("Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt")
```

You can also input Oscar polyhedra

```
using Oscar
using UniTriSat

P = cube(3)

triangulate(P)
```

Use Julia's package manager to run the test suite:

```
Pkg.test("UniTriSat")
```

See the function `triangulate` in `src/UniTriSat.jl` to see what options the function takes.

Options and return types
========
The triangulate function takes the following arguments:
### Basic Parameters & Flags

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`polytopes`** | | - | The target polytopes to be processed by the function. This can be a matrix encoding the vertices of the polytope, a Polyhedron object, a Vector containing multiple of the previous, an Oscar Polyhedron (or Vector thereof), or a path to a file. |
| **`intersection_backend`** | String | `"cpu"` | The computing backend to use (`"cpu"` or `"gpu"`). |
| **`unimodular`** | Boolean | `true` | Restricts the output to unimodular simplices. |
| **`regular`** | Boolean | `false` | Restricts the output to regular triangulations. |
| **`flag_triangulation`** | Boolean | `false` | Restricts the output to flag triangulations. |
| **`flag_SAT`** | Boolean | `false` | Experimental: Encodes flag-ness as additional clauses for the SAT-solver. |
| **`find_all`** | Boolean | `false` | Toggles whether to find all valid triangulations. |
| **`log_file`** | String | `""` | Path for logging (e.g., `"logs/my_run.log"`). Leave empty to disable. |
| **`terminal_output`** | String | `"final"` | Controls printing to the terminal. See below for details. |
| **`plot`** | Boolean | `false` | Toggles plotting. Projects to 3-faces in higher dimensions. Plots only the first result if `find_all` is `true`. |
| **`use_normaliz`** | Boolean | `false` | Uses Normaliz for interior lattice points (faster, but potentially unstable). |
| **`return_triangulations`** | String | `"first"` | Controls whether to return all, the first, or none of the found triangulations. See below for details. |
| **`solver`** | String | `"picosat"` | Decides the solver. `"picosat"` or `"cadical"` in general. `"d4"` for finding all solutions. |
| **`incremental_solving`** | Bool | `false` | Experimental: Use incremental solving. Only available with CaDiCaL. |
| **`parallel_split_solving`** | Bool | `true` | Enable parallel solving. |

### Output & Return Configurations

**`terminal_output`**
A string controlling what is printed to the terminal. It accepts any subset or combination of the following values:
* `"initial"`: Prints an initial summary of the run parameters.
* `"running"`: Prints intermediate results about the polytopes that are already done.
* `"table"`: Prints a table summarizing run times and memory usage split by operations (e.g., SAT solving etc.).
* `"final"`: Prints a final summary of the results.

**`return_triangulations`**
A string dictating what triangulations the function should return at the end of its run:
* `"all"`: Returns all found triangulations.
* `"first"`: Returns only the first found triangulation.
* `""` *(Empty string)*: Does not return any triangulations.

The results are returned as a RunResult struct which countains many TriangulationResult structs, as follows:
```
mutable struct TriangulationResult
    solution_simplices::Vector{Vector{Matrix{Int}}}
    number_of_triangulations_found::Int
    number_of_regular_triangulations_found::Int
    number_of_flag_triangulations_found::Int
    number_of_quadratic_triangulations_found::Int
    minimal_log::String
    total_time::Float64
    step_stats::Vector{StepStat}
end

mutable struct RunResult
    triangulation_results::Vector{TriangulationResult}
    number_triangulatable::Int
    number_regularly_triangulatable::Int
    number_flag_triangulatable::Int
    number_quadratic_triangulatable::Int
    total_number_of_triangulations_found::Int
    total_number_of_regular_triangulations_found::Int
    total_number_of_flag_triangulations_found::Int
    total_number_of_quadratic_triangulations_found::Int
    total_time::Float64
end
```

Tests
=============

After cloning the repository, you can run the test suite by doing `julia -t auto test.jl`, or you can test it in the package manager via `Pkg.test("UniTriSat")` in a `julia` terminal. It takes about 15 minutes on a laptop. 


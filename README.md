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

Tests
=============

You can run a set tests by doing `julia test.jl`.

We are in need of non-regular unimodular triangulations for our unit testing. If you know of any, other than the "mother of all examples" please write us. 


Getting started
===============

This package can be added as follows:

`import Pkg; Pkg.add(url="https://github.com/krhuang/UniTriSat/")`

You can add Polytope data to the directory Polytopes, e.g.

`cd Polytopes; git clone https://github.com/gabrieleballetti/small-lattice-polytopes`

Once this is done, you can run the code via the `triangulate` entry point function like so:

```
using UniTriSat

triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v6.txt")
```

See the function `triangulate` in `UniTriSat.jl` to see what options the function takes.

You can also run a set of correctness tests by doing `julia test.jl`.


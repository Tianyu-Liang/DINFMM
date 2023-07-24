# DINFMM
Code is all written in Julia, so you need to download the language from https://julialang.org/downloads/ before proceeding.
In addition, certain packages needs to be installed first before the code can be ran. For instance, near top of the file "kernel_2D.jl", there are many using statements, such as "using VectorizedRoutines". This means that the package VectorizedRoutines.jl needs to be added. to add a package, do:
```
username:> julia
julia> using Pkg
julia> Pkg.add("VectorizedRoutines")
```
If you are missing a package, the error message will tell you the corresponding package that is required.

The laplace folder contains the code for running a Laplace problem, while the helmholtz folder contains the code for running a Helmholtz example.

Each folder contains 3 examples: sequential_example512.jl, parallel_example512.jl, parallel_example2048.jl

Using Laplace as an example, to run the code, do:

```
username:> julia sequential_example512.jl
```

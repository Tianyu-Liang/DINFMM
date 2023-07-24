
t1 = time()
using Distributed, ClusterManagers
rmprocs(workers())
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
include("kernel_2D.jl")
using .kernel_2D
println("compile and addproc time: ", time() - t1)

factorize_and_solve(128^2, 4; run_solve=false) # set run_solve=true if you want to test solving a linear system after factorization 
factorize_and_solve(512^2, 6, 256; run_solve=false)


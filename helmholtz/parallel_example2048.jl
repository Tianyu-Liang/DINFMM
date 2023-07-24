using Distributed, ClusterManagers
rmprocs(workers())
addprocs(SlurmManager(4), exeflags=["--project=.", "--color=yes"], job_file_loc="newtest_4m4p256e2048new1")  # use this if on a system with slurm scheduler
# addprocs(4    # use this if you are using a machine without slurm
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
include("kernel_2D.jl")

using .kernel_2D
#println(BLAS.vendor())
factorize_and_solve(1024^2, 7; run_solve=false)
factorize_and_solve(2048^2, 8, 256; run_solve=false)

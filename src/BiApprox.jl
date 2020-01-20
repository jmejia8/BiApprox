module BiApprox

import LinearAlgebra: norm, dot
import MLKernels: Kernel, kernel, GaussianKernel, PeriodicKernel
import Random: randperm
import Statistics: mean, std, var


export kernel_approx_ul, approx_values, train_model, F̂
export train, train!, evaluate, KernelInterpolation, Data
export approximate

export LL_approx, bruteforce!, Ψ_approx, update_approximation_errors!, fit!

include("structures.jl")
include("kernelInterpolation.jl")
include("approximate.jl")
include("psi_approx.jl")
include("deprecated.jl")

end # module

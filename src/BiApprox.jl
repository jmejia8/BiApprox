module BiApprox

import LinearAlgebra: norm, dot
import MLKernels: Kernel, kernel, GaussianKernel


export kernel_approx_ul, approx_values, train_model, F̂
export train, train!, evaluate, KernelInterpolation, Data
export approximate

include("structures.jl")
include("kernelInterpolation.jl")
include("approximate.jl")
include("deprecated.jl")

end # module

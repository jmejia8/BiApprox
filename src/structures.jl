mutable struct Data
    Fs::Array{Float64, 1} # Fs = [ f(x) for x in X ]
    # X is a N×D matrix, that contains N row vectors in R^D
    Xs::Array{Float64, 2}
end

function Data(;Fs::Array{Float64, 1}=[], Xs::Array=Array{Float64, 2}[])
    
    sz = size(Xs)
    if length(sz) == 1
        return Data(reshape(Xs, sz[1], 1), Array{Float64, 1}(Fs))
    elseif length(sz) == 2
        return Data(reshape(Xs, sz[1], 1), Array{Float64, 1}(Fs))
    else
        @error "Xs shoud be a 2D Matrix and Fs a vector"
    end

    return nothing

end

function Data(F::Function, Xs::Array{Float64, 2})
    Fs = Float64[ F(Xs[i,:]) for i=1:size(Xs,1) ]
    Data(Fs, Xs)
end

function Data(F::Function, Xs::Array{Float64, 1})
    Fs = Float64[ F(Xs[i]) for i=1:length(Xs) ]
    Xs = reshape(Xs, length(Xs), 1)
    Data(Fs, Xs)
end





mutable struct KernelInterpolation
    kernel::Kernel
    trainingData::Data
    coeffs::Array{Float64}
    λ::Float64
end

function KernelInterpolation(trainingData::Data;kernel::Kernel = GaussianKernel(1.0), coeffs::Array = Array[], λ::Float64=0.0)
    KernelInterpolation(kernel, trainingData, coeffs, λ)
end

function KernelInterpolation(Fs::Array, Xs::Array; kernel::Kernel = GaussianKernel(1.0), coeffs::Array = Array[], λ::Float64=0.0)
    KernelInterpolation(kernel, Data(Fs, Xs), coeffs, λ)
end

function KernelInterpolation(F::Function, Xs::Array;kernel::Kernel = GaussianKernel(1.0), coeffs::Array = Array[], λ::Float64=0.0)
    KernelInterpolation(kernel, Data(F, Xs), coeffs, λ)
end
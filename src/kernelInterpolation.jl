function train!(method::KernelInterpolation)
    X = method.trainingData.Xs
    Fs = method.trainingData.Fs
    k = method.kernel

    N, D = size(X)

    A = ones(N+1, N+1)

    y = zeros(N+1)
    y[1:N] = Fs

    for i = 1:N
        for j = 1:N
            A[i, j] = kernel(k, X[i,:], X[j,:])
        end

        A[i, i] += method.λ
    end

    A[N+1, N+1] = 0.0

    # y = Ab
    # b = A⁻¹ ⋅ y
    
    b = inv(A) * y

    method.coeffs = b
    method
end

train(method::KernelInterpolation) = train!(method)

function evaluate(x::Real, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end

    X = method.trainingData.Xs
    k = method.kernel
    xx = Float64[x]
    K = [kernel(k,xx, X[i,:]) for i = 1:size(X,1)]

    dot( method.coeffs[1:end-1], K ) + method.coeffs[end]
end

function evaluate(x::Array{Float64, 1}, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end

    X = method.trainingData.Xs
    k = method.kernel
    K = [kernel(k,x, X[i,:]) for i = 1:size(X,1)]

    dot( method.coeffs[1:end-1], K ) + method.coeffs[end]
end

function evaluate(X::Array{Float64, 2}, method::KernelInterpolation)
    if length(method.coeffs) == 0
        @info("Training method...")
        train!(method)
    end

    Float64[ evaluate(X[i,:], method) for i=1:size(X,1) ]
end


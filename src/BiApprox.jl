module BiApprox

import LinearAlgebra: norm, dot
import MLKernels: Kernel, kernel, GaussianKernel

export kernel_approx_ul, approx_values, train_model, F̂

gaussKern(x::Vector, y::Vector; σ::Real=1.0) = MLKernels.kernel(GaussianKernel(σ), x, y)

function train_model(Fs::Vector, X::Matrix; k::Kernel = GaussianKernel(1.0), λ::Real = 0.0 )
    """
    X is a N×D, that contains N row vectors in R^D
    """

    N, D = size(X)

    A = ones(N+1, N+1)

    y = zeros(N+1)
    y[1:N] = Fs

    for i = 1:N
        for j = 1:N
            A[i, j] = kernel(k, X[i,:], X[j,:])
        end

        A[i, i] += λ
    end

    A[N+1, N+1] = 0.0

    # y = Ab
    # b = A⁻¹ ⋅ y
    
    b = inv(A) * y

    return b


end


F̂(x, b, X; k::Kernel = GaussianKernel) = dot( b[1:end-1], [kernel(k,x, X[i,:]) for i = 1:size(X,1)] ) + b[end]
approx_values(α::Vector, X::Matrix, X_data::Matrix; k::Kernel = GaussianKernel) = [F̂(X[i,:], α, X_data, k = k) for i = 1:size(X, 1)]

function kernel_approx_ul(Fs::Vector,
                     X_test::Matrix,
                    X_train::Matrix;
                     k::Kernel=GaussianKernel(1.0))

    """
    X is a N×D matrix, that contains N row vectors in R^D
    """

    α = train_model(Fs, X_train, k = k)

    return approx_values(α, X_test, X_train, k = k), α
    
end

end # module

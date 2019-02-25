module BiApprox

using LinearAlgebra

export kernel_approx_ul, approx_values, train_model, F̂

gaussKern(x::Vector, y::Vector; σ=1) = exp(- norm(x-y)^2 / (2σ^2) )

function train_model(Fs::Vector, X::Matrix, Y::Matrix; kernel::Function = gaussKern )
    """
    X is a N×D, that contains N row vectors in R^D
    Y like X
    """

    N, D = size(X)

    A = ones(N+1, N+1)

    y = zeros(N+1)
    y[1:N] = Fs

    for i = 1:N
        for j = 1:N
            A[i, j] = kernel(X[i,:], X[j,:])
        end
    end

    A[N+1, N+1] = 0.0

    # y = Ab
    # b = A'y
    
    b = inv(A) * y

    return b


end


F̂(x, b, X; kernel::Function = gaussKern) = dot( b[1:end-1], [kernel(x, X[i,:]) for i = 1:size(X,1)] ) + b[end]
approx_values(α::Vector, X::Matrix, X_data::Matrix; kernel::Function = gaussKern) = [F̂(X[i,:], α, X_data, kernel = kernel) for i = 1:size(X, 1)]

function kernel_approx_ul(Fs::Vector,
                     X_test::Matrix,
                    X_train::Matrix,
                    Y_train::Matrix;
                     kernel::Function=gaussKern)

    """
    X is a N×D matrix, that contains N row vectors in R^D
    Y like X
    """

    α = train_model(Fs, X_train, Y_train, kernel = kernel)

    return approx_values(α, X_test, X_train, kernel = kernel), α
    
end

end # module

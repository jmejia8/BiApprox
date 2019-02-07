module BiApprox

using LinearAlgebra

gaussKern(x, y; σ=1) = exp(- norm(x-y)^2 / (2σ^2) )

function train_model(F::Function, X::Matrix, Y::Matrix)
    """
    X is a N×D, that contains N row vectors in R^D
    Y like X
    """

    N, D = size(X)

    z = [ F(X[i,:], Y[i,:]) for i = 1:N ]

    K = [ Kernel( X[i,:], X[j,:] ) for i=1:N,j=1:N ]

    # z = Kα
    # α = K'z
    α = inv(K) * z


    F̂s = zeros(N)
    for i = 1:N
        F̂s[i] = F̂(X[i,:], α, X)
        @assert F̂s[i] ≈ z[i]
    end
    
    return α, F̂s, z
end


F̂(x, α, X) = dot( α, [Kernel(x, X[i,:]) for i = 1:size(X,1)]  )
approx_values(α::Vector, X::Vector, X_data::Matrix) = [F̂(X[i,:], α, X_data) for i = 1:size(X, 1)]

function kernel_approx_ul(F::Function,
                     X_test::Matrix,
                    X_train::Matrix,
                    Y_train::Matrix;
                     kernel::Function=gaussKern)

    """
    X is a N×D matrix, that contains N row vectors in R^D
    Y like X
    """

    α, F̂s, z = train_model(F, X_train, Y_train)

    return approx_values(α, X_test, X_train)
    
end

end # module

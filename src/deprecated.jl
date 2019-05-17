gaussKern(x::Array{Float64,1}, y::Array{Float64,1}; σ::Float64=1.0) = kernel(GaussianKernel(σ), x, y)

function train_model(Fs::Array{Float64,1}, X::Array{Float64,2}; k::Kernel = GaussianKernel(1.0), λ::Float64 = 0.0 )
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

function train_model(Fs::Array{Float64,1}, X_train::Array{Float64,1}; k::Kernel = GaussianKernel(1.0), λ::Float64 = 0.0 )
    train_model(Fs, reshape(X_train, length(X_train), 1), k = k)
end


F̂(x::Array{Float64,1}, b::Array{Float64,1}, X::Array{Float64}; k::Kernel = GaussianKernel()) = dot( b[1:end-1], [kernel(k,x, X[i,:]) for i = 1:size(X,1)] ) + b[end]


approx_values(α::Array{Float64,1}, X::Array{Float64,2}, X_train::Array{Float64,2}; k::Kernel = GaussianKernel()) = [F̂(X[i,:], α, X_train, k = k) for i = 1:size(X, 1)]

function kernel_approx_ul(Fs::Array{Float64,1},
                     X_test::Array{Float64,2},
                    X_train::Array{Float64,2};
                     k::Kernel=GaussianKernel(1.0))

    """
    X is a N×D matrix, that contains N row vectors in R^D
    """

    α = train_model(Fs, X_train, k = k)

    return approx_values(α, X_test, X_train, k = k), α
    
end

function kernel_approx_ul(Fs::Array{Float64,1},
                     X_test::Array{Float64,1},
                    X_train::Array{Float64,1};
                     k::Kernel=GaussianKernel(1.0))

    """
    X is a N×1 array, that contains N row vectors in R^D
    """



    return kernel_approx_ul(Fs, reshape(X_test, length(X_test), 1), reshape(X_train, length(X_train), 1), k=k)
    
end
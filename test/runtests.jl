using Test
using BiApprox

function test1()
    L(w, a, b) = a + w^2 - a*cos( π * b * w)
    f(x,y; n=length(y)-1) = abs( 4x[n+1] - x[n+2] ) * y[end]^2 + sum(L.( abs.(x[1:n]-y[1:n]), x[n+1], x[n+2]))
    F(x,y; n=length(y)-1) = abs( 4x[n+1] - x[n+2] ) / (1+y[end]^2) + sum(L.( abs.(x[1:n]-y[1:n]), x[n+1], x[n+2]))

    Ψ(X::Matrix) = begin
        n = size(X,2)-2
        Y = zeros(N, n+1)
        Y[:,1:n] = X[:,1:n]
        Y        
    end

    n = 5
    N = 300
    D_ul = n+2
    D_ll = n+1

    X = -10 .+ 20rand(N, D_ul)
    X[:, n+1:n+2] = 2 .+ 8rand(N, 2)

    Y = Ψ(X)

    Fs =  [ F(X[i,:], Y[i,:]) for i =1:N ]


    X_test = -10 .+ 20rand(N, D_ul)
    X_test[:, n+1:n+2] = 2 .+ 8rand(N, 2)
    Y_test = Ψ(X_test)
    Fs_test =  [ F(X_test[i,:], Y_test[i,:]) for i =1:N ]
    

    F̂s_test, α = kernel_approx_ul(Fs,X_test, X)

    true


end

function test2()
    f(x) = sum(x.^2)
    N = 100
    D = 11
    X = randn(N, D)

    method = KernelInterpolation(f, X)
    train!(method)

    X_test = rand(97, D)

    ff = evaluate(X_test, method)

    method2 = KernelInterpolation(f, X)
    train!(method2)
    ff2 = [evaluate(X_test[i,:], method2) for i = 1:size(X_test, 1)]


    return sum(abs.(ff - ff2)) ≈ 0.0


end

function test3()
    f(x) = sum(x.^2)
    N = 100
    D = 11
    X = randn(N, D)
    X2 = randn(N, D)

    model = KernelInterpolation(f, X; λ = 0.1)
    train!(model)
    f_approx = approximate(model)

    model2 = KernelInterpolation(f, X2; λ = 0.1)
    train!(model2)
    f_approx2 = approximate(model)

    X_test = rand(97, D)

    ff = [f_approx( X_test[i,:] ) for i = 1:15 ]
    ff2 = [f_approx2( X_test[i,:] ) for i = 1:15 ]

    return sum(abs.(ff - ff2)) ≈ 0.0

end

function test4()
    Ψ(x) = [ sum(x.^2), x[1] - x[2].^2, sin(x[3]), abs(x[1]) ]

    X = -10 .+ 20rand(100, 3)
    Y = zeros(100, 4)
    
    for i = 1:size(Y,1)
        Y[i,:] = Ψ(X[i,:])
    end

    model = LL_approx(X, Y, p_train=0.8, scale_values=0.1ones(3))

    bruteforce!(model, debug=false)

    sum(model.approx_errors) >= 0

end

@test test1()
@test test2()
@test test3()
@test test4()
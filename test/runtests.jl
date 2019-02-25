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
    

    F̂s_test, α = kernel_approx_ul(Fs,X_test, X,Y)

    println(sum(abs.(Fs_test- F̂s_test))/N)

    true


end

@test test1()
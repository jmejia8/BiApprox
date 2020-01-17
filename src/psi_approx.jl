mutable struct LL_approx{T}
    models::Vector{T}
    λ::Vector{Float64}
    indexes::Matrix{Bool}
    
    X_train::Matrix{Float64}
    Y_train::Matrix{Float64}

    X_test::Matrix{Float64}
    Y_test::Matrix{Float64}

    ys::Vector{Function}

    approx_error::Float64

end

function LL_approx( X_train,
                    Y_train;
                    models = nothing,
                    λ = 1e-5,
                    indexes = zeros(Bool, 0, 0),
                    X_test  = zeros(Bool, 0, 0),
                    Y_test  = zeros(Bool, 0, 0),
                    ys = Function[],
                    p_train = 0.8,
                    approx_error = 0.0)
    

    # TODO
    LL_approx(models, λ, indexes, X_train, Y_train, X_test, Y_test, ys, approx_error)
end

function gen_y_approx(X, Y; indexes = ones(Bool, size(X, 2)), λ=1e-2)

    Yx = Y

    if sum(indexes) == 0 || var(Yx) < 1e-8
        return x -> mean(Yx)
    end

    Xtr = X[:, indexes]
    model = KernelInterpolation(Yx, Xtr, λ=λ, kernel=PeriodicKernel())
    train!(model)
    approximate(model) # returns a function ŷ_i
end

function get_λ(X, Y, Xts, Yts, indexes)

    N = length(Yts)


    ff(λ) = begin
        y = gen_y_approx(X, Y, indexes = indexes, λ = λ[1])

        s = 0.0
        for i = 1:N
            y_approx = Ψ_approx(view(Xts, i,:), y, indexes )
            s += (y_approx - Yts[i])^2
        end
        sqrt(s)
    end

    # res = Metaheuristics.optimize(ff, Array([0.0 1]'), ECA(N= 19,options= Metaheuristics.Options(f_calls_limit=19, debug=false)))
    # return res.best_sol.x[1], res.best_sol.f
    
    # res.best_sol.f = ff([1e-5])
    # res.best_sol.x = [1e-5]
 vb     return 1e-5, ff([1e-5])
end

Ψ_approx(x, y, indexes) = y(x[indexes])

function Ψ_approx(x, ys, indexes) 
    y = zeros(length(ys))

    for i = 1:length(ys)
        y[i] = ys[i](x[view(indexes, i,:)])
    end
    y
end

function get_indexes(X, Y, Xts, Yts)
    N, D_ll = size(Yts)
    D_ul = size(Xts, 2)

    
    indexes = zeros(Bool, size(Y, 2), size(X, 2))
    ress = []
    for i = 1:D_ll
        @show i
        ff(indexes) = get_λ(X, Y[:,i], Xts, Yts[:,i], indexes)

        
        if D_ul > 10
            indexes_initial = rand(Bool[0, 1], size(X, 2))
            indexes_new, res = hillClimbing(ff, indexes_initial, iters=100, debug=true, r = 2)
        else
            indexes_new, res = bruteforce(ff, D_ul)
        end

        indexes[i, :] =  indexes_new
        push!(ress, res)        
        
        @show indexes_new
    end
    
    indexes, ress

end

function Ψ_approx(X_train, Y_train, X_test, Y_test)
    ys = Function[]
    
    for i = 1:D_ll
        y = gen_y_approx(X_test, view(Y_test, :, i), indexes=view(indexes, :, i), λ = λ)
        push!(ys, y)
    end

    y
end



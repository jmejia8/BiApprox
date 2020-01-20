mutable struct LL_approx
    models::Vector{AbstractModel}
    λ::Vector{Float64}
    indexes::Matrix{Bool}
    
    X_train::Matrix{Float64}
    Y_train::Matrix{Float64}

    X_test::Matrix{Float64}
    Y_test::Matrix{Float64}

    ys::Vector{Function}

    approx_errors::Vector{Float64}
    scale_values::Matrix{Float64}

end

function LL_approx( X_train,
                    Y_train;
                    models = AbstractModel[],
                    λ = [1e-5 for i=1:size(Y_train, 2)],
                    indexes = zeros(Bool, 0, 0),
                    X_test  = zeros(Float64, 0, 0),
                    Y_test  = zeros(Float64, 0, 0),
                    ys = Function[],
                    p_train = 0.8,
                    approx_errors = [Inf],
                    scale_values = ones(1,size(X_train, 2)))
    
    if size(X_train, 1) != size(Y_train, 1)
        @error("Check size of the train set (X_train, Y_train)")
    end
    if length(size(scale_values)) == 1
        scale_values = Array(scale_values')
    end

    X_train = X_train .* scale_values

    if (isempty(X_test) || isempty(Y_test))
        n = round(Int, p_train*size(X_train, 1))
        idx = randperm(size(X_train, 1))

        X_test = X_train[idx[n+1:end], :]
        Y_test = Y_train[idx[n+1:end], :]

        X_train = X_train[idx[1:n], :]
        Y_train = Y_train[idx[1:n], :]
    else
        X_test = X_test .* scale_values
    end

    LL_approx(models, λ, indexes, X_train, Y_train, X_test, Y_test, ys,
                approx_errors, scale_values)
end


function fit!(ll_approx::LL_approx)
    empty!(ll_approx.models)
    empty!(ll_approx.ys)

    for i = 1:size(ll_approx.Y_train, 2)
        Yx = view(ll_approx.Y_train, :, i)
        if sum(ll_approx.indexes[i,:]) == 0 || var(Yx) < 1e-8
            y =  x -> mean(Yx)
            model = KernelInterpolation(zeros(1), zeros(1,1))
        else
            Xtr = view(ll_approx.X_train, :, ll_approx.indexes[i,:])
            model = KernelInterpolation(Yx, Xtr, λ = ll_approx.λ[i], kernel=PeriodicKernel())
            train!(model)
            y = approximate(model)

        end
        # approximate(m) returns a function ŷ_i
        push!(ll_approx.models, model)
        push!(ll_approx.ys, y)

    end
end


function update_approximation_errors!(ll_approx)
    empty!(ll_approx.approx_errors)


    s = zeros(size(ll_approx.Y_train, 2))
    for i = 1:size(ll_approx.X_test, 1)
        y_approx = Ψ_approx(view(ll_approx.X_test, i,:), ll_approx, false)
        s += (y_approx - view(ll_approx.Y_test, i, :)).^2
    end

    ll_approx.approx_errors = sqrt.(s)    

end


function optimize_indexes!(model::LL_approx, debug=false)
    # D_ul = size(model.X_test, 1)
    # if D_ul > 10
        # indexes_initial = rand(Bool[0, 1], size(X_train, 2))
        # indexes_new, res = hillClimbing(ff, indexes_initial, iters=100, debug=true, r = 2)
    # else
    # end
    bruteforce!(model, debug=debug)

end

function Ψ_approx(x, ll_approx, scale_data=true) 
    y = zeros(length(ll_approx.ys))

    if scale_data
        x = x .* scale_values
    end
    # x = xx[1,:]

    for i = 1:length(ll_approx.ys)
        y[i] = ll_approx.ys[i](x[view(ll_approx.indexes, i,:)])
    end
    y
end

function bruteforce!(ll_approx; debug=false)
    D = size(ll_approx.X_train, 2)
    D_ll = size(ll_approx.Y_train, 2)

    approx_errors_old = Inf*ones(D_ll)
    indexes_best = zeros(D_ll, D)

    for t = 1:2^D
        x = parse.(Int, collect(bitstring(t)), base=10)
        reverse!(x)

        x_new = Bool.(x[1:D])
        indexes = repeat(x_new', D_ll)


        ll_approx.indexes = indexes

        fit!(ll_approx)
        update_approximation_errors!(ll_approx)


        for i = 1:D_ll
            if ll_approx.approx_errors[i] < approx_errors_old[i]
                approx_errors_old[i] = ll_approx.approx_errors[i]
                indexes_best[i,:] = indexes[i,:]
            end
        end

        if debug
            println(">>>>>>>>>>>> iter: ", t)
            display(indexes_best)
            println("err: ", approx_errors_old)
        end

    end

    ll_approx.indexes = indexes_best

    fit!(ll_approx)
    update_approximation_errors!(ll_approx)
    

    indexes_best
end

    
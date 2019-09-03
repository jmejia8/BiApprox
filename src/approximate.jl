function approximate(f, X)
    method = KernelInterpolation(f, X)
    train!(method)
    return x -> evaluate(x, method)
end

function approximate(method)
    if length(method.coeffs) == 0
        train!(method)
    end

    return x -> evaluate(x, method)
end
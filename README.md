# Approximate Methods

## Kernel Interpolation

```julia
# exact model
f(x) = sum(x.^2)

N = 100
D = 11
X = randn(N, D) # training data

# Method to approximate f
method = KernelInterpolation(f, X)

# test data
X_test = rand(97, D)

# approximated values
ff = evaluate(X_test, method)
```
using Test
using BiApprox

function test1()
    L(w, a, b) = a + w^2 - a*cos( π * b * w)
    f(x,y; n=length(y)-1) = abs( 4x[n+1] - x[n+2] ) * y[end]^2 + sum(L.( abs.(x[1:n]-y[1:n]), x[n+1], x[n+2]))
    F(x,y; n=length(y)-1) = abs( 4x[n+1] - x[n+2] ) / (1+y[end]^2) + sum(L.( abs.(x[1:n]-y[1:n]), x[n+1], x[n+2]))

    n = 10
    x = 10rand(n+2)
    y = zeros(n+1)

    y[1:n] = x[1:n]

    # println(F(x,y))
    q1 = f(x,y) 

    x[n+2] = 4x[n+1]

    q2 = F(x,y) 

    println(q1, " ", q2 )

    q2 + q1 ≈ 0    


end

@test test1()
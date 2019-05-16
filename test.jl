using BiApprox
using CEC17
using Plots
plotly(legend=false)

function test_2d()
	N = 200
	X_train = -100 .+ 200rand(N, 2) 

	X = Y = range(-100, 100, length = 100)
	f(x) = cec17_test_func(x, 5) #sum(x.^2)

	fs = [ f(X_train[i,:]) for i=1:size(X_train, 1) ]

	a = train_model(fs, X_train, λ=3.5)

	plot(title="Plot")
	surface(X, Y, (x,y)->f([x,y]))
	scatter!(X_train[:,1], X_train[:,2], fs; markersize=1)
	surface!(X, Y, (x,y)->F̂([x,y], a, X_train))
end

function test_1d()
	N = 200
	X_train = -5 .+ 10rand(N) 

	X = range(-5, 5, length = 100)
	f(x) = sum(sin.(x))

	fs = [ f(X_train[i,:]) for i=1:length(X_train) ]

	ff, a = kernel_approx_ul(fs, X, X_train, λ=0.5)

	plot(title="Plot")
	plot(X, f.(X))
	scatter!(X_train, fs; markersize=1)
	plot!(X, ff)
end

test_2d()


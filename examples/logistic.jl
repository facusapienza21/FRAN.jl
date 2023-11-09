"""
turing-example.jl
"""

using Pkg
Pkg.activate(".")

using Random 
using Turing, TuringGLM
using StatsModels
using StatsPlots
using DataFrames
# We need a logistic function, which is provided by StatsFuns.
using StatsFuns: logistic
using Optim 
using LinearAlgebra: I

p = 5          # Number of features 
q = 100
n = 20 * q     # Number of points

# set seed 
rng = MersenneTwister(616) 


### Generate synthetic data and run simple Logistic Regression 

# df = DataFrame(rand(rng, Normal(), (n,p)), :auto)
# # Increase relevance of one of the covariates
# df."x3" *= 3
# df."y" = rand(rng, Bernoulli(0.5), n)

X = rand(rng, Normal(), (n,p))
Y = [rand(rng, Bernoulli(logistic(2*X[i,1]))) for i in 1:n];

# Let's first fit a simple logistic regression model 
# fm = @formula(y ~ 1 + x1 + x2 + x3)
# fm = term(:y) ~ sum(term.([1; names(df, Not(:y))]))

run_turingGLM = false

if run_turingGLM
    model = turing_model(fm, df; model=Bernoulli)

    chn = sample(model, NUTS(), 1000)

    fig = plot(chn)
    savefig(fig, "chain.png")
end

### Now, let's just fit the Logistic regression model and add noise using FRAN

# Define a simple logistic model by hand

@model function logistic_regression(x, y, n, p, σ)

    # intercept ~ Normal(0, σ)
    # slope1 ~ Normal(0, σ)
    # slope2 ~ Normal(0, σ)
    slopes ~ MvNormal(zeros(p), σ^2 * I)

    for i in 1:n
        prob = logistic(sum(slopes .* x[i,:]))
        # prob = logistic(intercept + slope1 * x[i,1] + slope2 * x[i,2])
        y[i] ~ Bernoulli(prob)
    end

end

# X = reduce(hcat, (df."x1", df."x2"))

model = logistic_regression(X, Y, n, p, 1.0)

# chain = sample(model, HMC(0.05, 10), MCMCThreads(), 10_000, 3)

# We can compare with the maximum likelihood
mle_estimate = optimize(model, MLE())

# We can also sample directly from the prior
# sample(model, Prior(), 1000)


# Customized sampler

include("../src/sample/LogisticSampler.jl")

θ₀ = zeros(p)
spl_logistic = LogisticSampler(θ₀)

chain = sample(model, spl_logistic, 1_000)
"""
turing-example.jl
"""

using Pkg
Pkg.activate("../FRAN")

using Random 
using Turing, TuringGLM
using StatsPlots
using DataFrames

p = 100         # Number of features 
q = 100
n = 200 * q     # Number of points

# set seed 
rng = MersenneTwister(616) 


### Generate synthetic data and run simple Logistic Regression 

df = DataFrame(rand(rng, Normal(), (n,p)), :auto)
# Increase relevance of one of the covariates
df."x3" *= 3
df."y" = rand(rng, Bernoulli(0.5), n)
# df."X" = X

# Let's first fit a simple logistic regression model 
fm = @formula(y ~ 1 + x1 + x2 + x3)

model = turing_model(fm, df; model=Bernoulli)

chn = sample(model, NUTS(), 100)

fig = plot(chn)
savefig(fig, "chain.png")


### Now, let's just fit the Logistic regression model and add noise using FRAN


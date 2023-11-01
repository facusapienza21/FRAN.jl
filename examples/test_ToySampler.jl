include("../src/sample/Toy.jl")

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5, 3), 30)

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = ToySampler([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 1000)
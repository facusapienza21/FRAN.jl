import AbstractMCMC
using Distributions
using Random
using LinearAlgebra: I

using Infiltrator

struct ToySampler{T, D} <: AbstractMCMC.AbstractSampler 
    init_θ  ::T     # Initial parameter of the prior
    proposal::D     # Distribution of the prior
end

# Default contructor
ToySampler(init_θ::Real) = ToySampler(init_θ, Normal(0,1))
ToySampler(init_θ::Vector{<:Real}) = ToySampler(init_θ, MvNormal(zero(init_θ), I))

# Define a model type. Stores the log density function.
struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
    log_likelihood::F
end

# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T, L}
    θ               ::T
    log_likelihood  ::L
end

# Store the new draw and its log density.
Transition(model::DensityModel, θ) = Transition(θ, log_likelihood(model, θ))

# Define a log-likelihood model 
log_likelihood(model::DensityModel, θ) = model.log_likelihood(θ)
log_likelihood(model::DensityModel, t::Transition) = t.log_likelihood(θ)

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DensityModel,
    spl::ToySampler,
    N::Integer,
    ::Nothing;
    kwargs...
)
    return Transition(model, spl.init_θ)
end

# Define the other step function. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DensityModel,
    spl::ToySampler,
    ::Integer,
    θ_prev::Transition;
    kwargs...
)
    # Let's just return a perturbed version of θ
    return Transition(model, spl.init_θ .+ rand(Normal()))
end
import AbstractMCMC
using Distributions
using Random
using LinearAlgebra: I

using Infiltrator

"""
Define a model type. Stores the log density function.
""" 
struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
    log_likelihood::F
end

"""
Create a very basic Transition type, only stores the 
parameter draws and the log probability of the draw.
"""
struct Transition{T, L}
    θ               ::T
    log_likelihood  ::L
end
Transition(model::DensityModel, θ) = Transition(θ, log_likelihood(model, θ))

"""
Basic state to storage
"""
struct State{
    TTrans<:Transition
}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
end

"""
Toy sampler
"""
struct ToySampler{T, D} <: AbstractMCMC.AbstractSampler 
    init_θ  ::T     # Initial parameter of the prior
    proposal::D     # Distribution of the prior
end
ToySampler(init_θ::Real) = ToySampler(init_θ, Normal(0,1))
ToySampler(init_θ::Vector{<:Real}) = ToySampler(init_θ, MvNormal(zero(init_θ), I))



"""
Step initializer
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DensityModel,
    spl::AbstractMCMC.AbstractSampler;
    initial_params = nothing,
    init_params = initial_params,
    kwargs...,
)   
    println("Initializing iterator...")

    t = Transition(model, spl.init_θ)
    state = State(0, t)
    return AbstractMCMC.step(rng, model, spl, state)
end

"""
Step iterator
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DensityModel,
    spl::AbstractMCMC.AbstractSampler,
    state::State;
    kwargs...,
)
    # Compute transition.
    i = state.i + 1
    t_old = state.transition
    θ_old = t_old.θ

    θ_new = θ_old .+ rand(Normal(), 2)

    # Transition 
    t_new = Transition(model, θ_new)
    state_new = State(i, t_new)

    return θ_new, state_new
end


# Define a log-likelihood model 
log_likelihood(model::DensityModel, θ) = model.log_likelihood(θ)
log_likelihood(model::DensityModel, t::Transition) = t.log_likelihood(θ)
import AbstractMCMC
using Distributions
using Random
using LinearAlgebra: I, Diagonal, inv

using Infiltrator

"""
Least square estimator. This can be done much better but for now it does the job.
"""
function OLS(X::Matrix{<:Real}, b::Vector{<:Real}, W::Vector{<:Real})
    return inv(X' * Diagonal(W) * X) * X' * Diagonal(W) * b
end

# """
# Define a model type. Stores the log density function.
# """ 
# struct DensityModel{F<:Function} <: AbstractMCMC.AbstractModel
#     log_likelihood::F
# end

"""
Create a very basic Transition type, only stores the 
parameter draws and the log probability of the draw.
"""
struct Transition{T, L}
    θ               ::T
    log_likelihood  ::L
end
# Transition(model::AbstractMCMC.AbstractModel, θ) = Transition(θ, log_likelihood(model, θ))
Transition(model::AbstractMCMC.AbstractModel, θ) = Transition(θ, log_likelihood(model, θ))

"""
Basic state to storage
"""
struct LogisticState{
    TTrans<:Transition
}
    "Index of current iteration."
    i::Int
    "Current [`Transition`](@ref)."
    transition::TTrans
    "Current η = Xθ"
    η::Vector{<:Real}
    # "Pseudo-response"
    # Z::Vector{<:Real}
    # "Newton weights (diagonal)"
    # # W::Matrix{<:Real}
    # W::Vector{<:Real}
end

"""
Logistic sampler
"""
struct LogisticSampler{T} <: AbstractMCMC.AbstractSampler 
    init_θ  ::T             # Initial parameter of the prior
    step::AbstractFloat     # Step of resampled model
end
LogisticSampler(init_θ::Real) = LogisticSampler(init_θ, 1.0)
LogisticSampler(init_θ::Vector{<:Real}) = LogisticSampler(init_θ, 1.0)


"""
Step initializer
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler;
    initial_params = nothing,
    init_params = initial_params,
    kwargs...,
)   
    println("Initializing iterator...")
    t = Transition(model, spl.init_θ)
    state = LogisticState(0, t, model.args.x * spl.init_θ)
    return AbstractMCMC.step(rng, model, spl, state)
end

"""
Step iterator
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler,
    state::LogisticState;
    kwargs...,
)
    # Mean response
    μ = logistic.(state.η)
    newton_weights = μ .* (1 .- μ)

    # Sample from forward model 
    # To do: find a way to sample forward with Turing by fixing the parameters
    pseudo_y = [rand(Bernoulli(prob)) for prob in μ]

    # pseudo_response = state.η + (model.args.y - μ)./newton_weights    
    pseudo_response = state.η + (model.args.y - pseudo_y)./newton_weights    

    # Compute transition.
    # Solve the least square problem pseudo_response ≈ Xβ with weights
    θ_new = OLS(model.args.x, pseudo_response, newton_weights)

    # Define new state variable
    t = Transition(model, θ_new)
    state_new = LogisticState(state.i+1, t, model.args.x * θ_new)

    return θ_new, state_new
end


# Define a log-likelihood model
function log_likelihood(model::DynamicPPL.Model, θ) 
    # println("define model with DynamicPPL.Model")
    f = LogDensityFunction(model)
    # Sign of the log likelihood?
    return DynamicPPL.LogDensityProblems.logdensity(f, θ)
end
# log_likelihood(model::DensityModel, t::Transition) = t.log_likelihood(θ)
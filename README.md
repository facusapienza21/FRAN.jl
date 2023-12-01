### FRAN sampler based on Turing.jl

The package `Turing.jl` (and its child dependencies) allows to define a customized sampler. As the [documentation](https://turinglang.org/v0.29/docs/using-turing/external-samplers) indicates, this is done by defining a new type of `AbstractMCMC.step`, both for the initialization of the chain and the iteration of the sampler. 
The same one is defined as 
```julia
# Initilization 
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler;
    initial_params = nothing,
    init_params = initial_params,
    kwargs...,
)   
	...
	return AbstractMCMC.step(rng, model, spl, state)
end

# Iteration 
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AbstractMCMC.AbstractSampler,
    state::LogisticState;
    kwargs...,
)
	...
	return θ_new, state_new
end
```

Notice that here we are not following the convention explained in the documentation... instead, this is more similar to what is in the [implementation of AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl/blob/master/src/abstractmcmc.jl). We can instead return a `Transition` type as first argument, but returning the new paramter also makes the work. 

We further need to define 
- model: This consist of the probabilistic model we are considering and it is usually defined independenlty of the sampler. It has to be a subtype of `AbstractMCMC.AbstractModel`
- spl: Consist of the new object for the new sampler. It should include the parameters of the sampler, like initialization, parameters controling step updates. These are the hyper-parameters of the chain. This has to be a subtype of `AbstractMCMC.AbstractSampler`.
- state: Customized structure to include all required information during the iteration step. 

Notice that the customized `state` has an argument known as `Transition`. This is another type that needs to be defined and includes basic information on the parameter and probabilistic model. It's just a macro to return the new sample at each iteration of the sampler with some extra information, such as the log-likelihood of the model at that time. 
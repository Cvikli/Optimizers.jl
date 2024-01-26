module Optimizers


using Aritmetics: ⌂, ϵ64, rel_stable_diff
using Pythonish
using Boilerplate
using CUDA

abstract type AbstractOptimiser end
abstract type AbstractTrainingOptimiser end

batch!(Sch, ::AbstractOptimiser) = nothing
apply!(o::AbstractOptimiser, Δ) = @assert false "TODO extend"
apply®!(o::AbstractOptimiser, Δ) = @assert false "we already handle state differently... please be noted that this algorithm can also implement this to handle it specially." 
update_opt_train!(o::AbstractTrainingOptimiser, lossₙ, lossₙ₋₁) = @assert false "TODO extend"
validate(o::AbstractTrainingOptimiser) = nothing

@inline default_loss() = ⌂
@inline default_lr() = 2f-3
@inline default_v() = 1f0
@inline default_g() = 0f0
@inline default_®v() = 1f0
@inline default_®g() = 0f0
@inline default_min() = nextfloat(typemin(Float32))
@inline default_max() = prevfloat(typemax(Float32))
@inline default_Δloss_up() = 1f-10

# lr_rate_update_fn(Δloss, Δloss_up) = Δloss <= 1.02f0 ? (Δloss_up < 8.0f0 ? 1.6323f0 : Δloss_up < 1024.0f0 ? 1.17f0 : Δloss_up < 1098304.0f0 ? 1.07f0 : 1.04f0) : 0.05691517f0
# @inline lr_rate_update_fn(Δloss, Δloss_up) = Δloss <= 1.0001f0 ? 2.18917 : Δloss < 1.007f0 ? 0.03957f0 : Δloss < 1.3f0 ? 0.000389f0 : 0.0000000489f0#* Δloss
@inline lr_rate_update_fn(Δloss, Δloss_up) = Δloss < 0.98f0 ? 1.0265224453f0 : Δloss < 1.0001f0 ? 1.018917f0 : Δloss < 1.07f0 ? 0.247213f0 : Δloss < 1.4f0 ? 0.00015137f0 : 6.3134103f-6 

@inline lr_rate_update_fn_generator(loss_window) = (let lw = loss_window; p1 = 1.028917f0;  
"Δloss < 0.98f0 ? $(p1 ^ 4.1792f0)f0 : Δloss < 1.0001f0 ? $p1 : Δloss < 1.07f0 ? $(Float32(1f0 / (p1 ^ (0.714f0 * lw))))f0 : Δloss < 1.4f0 ? $(Float32(1f0 / (p1 ^ (2.057f0*lw))))f0 : $(Float32(1f0 / (p1 ^ (2.8f0 * lw))))f0"
end)  # PRM: C1, C2, C3, C4, P1, Speedup1, SlowDown1, SlowDown2, SlowDown3
# TODO: should use weak refs

"""
    Descent(η = 0.1)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.

# Examples
```julia
opt = Descent()

opt = Descent(0.3)

ps = Flux.params(model)

gs = gradient(ps) do
    loss(x, y)
end

Flux.Optimise.update!(opt, ps, gs)
```
"""
mutable struct Descent <: AbstractOptimiser
  eta::Float64
end


Descent() = Descent(0.1)

function apply!(o::Descent, x, Δ)
  Δ .*= o.eta
end

# Example optimizer api.
mutable struct MarciOptimizer{CPU_FLOAT_3, DEV_FLOAT_3} <: AbstractOptimiser
  eta::Float32  
	p_ids::Vector{Int64}
	®_ids::Vector{Int64}
	v::DEV_FLOAT_3
	g::DEV_FLOAT_3
  ∑®v::CPU_FLOAT_3
  ∑®g::CPU_FLOAT_3
  ®v::DEV_FLOAT_3
  ®g::DEV_FLOAT_3
end


MarciOptimizer(B, ∑B, p_ids, ®_ids, η::Real=1f0) = MarciOptimizer(η, 
                                                      p_ids,
                                                      ®_ids,
                                                      CUDA.fill(default_v(),  1,  1, len(p_ids)),
                                                      CUDA.fill(default_g(),  1,  1, len(p_ids)),
                                                            fill(default_®v(), ∑B, 1, len(®_ids)),
                                                            fill(default_®g(), ∑B, 1, len(®_ids)),
                                                      CUDA.fill(default_®v(), B,  1, len(®_ids)),
                                                      CUDA.fill(default_®g(), B,  1, len(®_ids)),)
MarciOptimizer(a, B, ∑B, η::Real=1f0) = MarciOptimizer(B, ∑B, a.p_ids, a.®_ids, η)
# function batch_opt_save!(Sch, o::MarciOptimizer)
  # slice = current_slice(Sch)
function batch_opt_save!(slice, o::MarciOptimizer)
  o.∑®v[slice, :, :] = Array(o.®v)
  o.∑®g[slice, :, :] = Array(o.®g)
end
# function batch!(Sch, o::MarciOptimizer)
  # slice = current_slice(Sch)
function batch!(slice, o::MarciOptimizer)
  o.®v = CuArray(o.∑®v[slice, :, :])
  o.®g = CuArray(o.∑®g[slice, :, :])
end
alpha_momentum(Δ, v, g) = min(ifelse(g * Δ >= 0f0, v *1.037f0, 1f0), 1f4)
function apply®!(o::MarciOptimizer, Δ)
  η = o.eta
  @. o.®v = alpha_momentum(o.®v, o.®g)
  @. o.®g = Δ
  @. Δ *= η * o.®v
end
apply!(o::MarciOptimizer, x, Δ, idxs) = apply!(o, Δ)
function apply!(o::MarciOptimizer, Δ)
  η = o.eta
  @. o.v = alpha_momentum(Δ, o.v, o.g)
  @. o.g = Δ
  @. Δ *= η * o.v
end

mutable struct TrainingOptimiser <: AbstractTrainingOptimiser
  opt_lr::Float32
  opt_max_lr::Float32
  opt_Δloss_min::Float32 
  opt_Δloss_up::Float32
  opt_Δlr_decrease::Float32
end
TrainingOptimiser() = TrainingOptimiser(default_lr(), default_lr(), default_min(), default_Δloss_up(), 1f0)
apply!(o::TrainingOptimiser, Δ) = @. Δ *= o.opt_lr

converging(trg, v, α) = (p= 0.01f0; return p*trg*(α*1.013) + (1f0-p)v) 

update_opt_train!(o::TrainingOptimiser, lossₙ, rel_lossₙ, ©loss, ©rel_loss, loop_window) = begin
  opt_Δloss_min, opt_lr, opt_max_lr, opt_Δlr_decrease = o.opt_Δloss_min, o.opt_lr, o.opt_max_lr, o.opt_Δlr_decrease
	o.opt_Δloss_min = min(lossₙ, opt_Δloss_min)
  ©loss == ⌂ && return
  o.opt_Δloss_up += (isfinite(lossₙ) && isfinite(©loss) && ©loss < lossₙ) ? lossₙ - ©loss : 0.0f0

  if opt_lr < (α=0.33) * opt_max_lr
    if opt_Δlr_decrease < 0.36
      opt_lr = converging(opt_max_lr, opt_lr, α*0.9)
    else
      opt_lr = converging(opt_max_lr, opt_lr, α)
    end
  # elseif opt_lr < (α=0.63) * opt_max_lr
    # opt_lr = converging(opt_max_lr, opt_lr, α)
  else
    Δloss = rel_stable_diff(lossₙ, ©loss)
    relΔloss = rel_stable_diff(rel_lossₙ, ©rel_loss)
    if Δloss < 1.0001f0#.07
      lr_mul = lr_rate_update_fn(Δloss, o.opt_Δloss_up)
    else
      lr_mul = lr_rate_update_fn(relΔloss, o.opt_Δloss_up)
    end
    opt_lr *= lr_mul
  end
  opt_lr = min(2^22, max(opt_max_lr*1f-8, opt_lr))
  o.opt_Δlr_decrease *= min(1f0,opt_lr/o.opt_lr)
	o.opt_lr = opt_lr
end
validate(o::TrainingOptimiser) = o.opt_lr < 1f-26 && @warn "$o learning rate is dangerously too low"

mutable struct IdentityTrainingOptimiser <: AbstractTrainingOptimiser
end
apply!(o::IdentityTrainingOptimiser, Δ) = @. Δ *= 1f-1
update_opt_train!(o::IdentityTrainingOptimiser, lossₙ, rel_lossₙ, ©loss, ©rel_loss,  loop_window) = nothing


mutable struct TrainingBatchOptimiser
  opt_lr::Vector{Float32}
  opt_Δloss_min::Vector{Float32}
  opt_Δloss_up::Vector{Float32}
end
apply!(o::TrainingBatchOptimiser, slice, Δ) = @. Δ *= o.opt_lr[slice]
update_opt_train!(o::TrainingBatchOptimiser, slice, lossₙ, lossₙ₋ₜ) = begin
  opt_Δloss_min, opt_lr = o.opt_Δloss_min[slice], o.opt_lr[slice] 
  # lossₙ, lossₙ₋ₜ = Array(lossₙ), Array(lossₙ₋ₜ) # TODO Marci optimise this was on CUDA!
	@. o.opt_Δloss_min[slice] = min(lossₙ, opt_Δloss_min)
  lossₙ₋ₜ == ⌂ && return
  @. o.opt_Δloss_up[slice] += ifelse(isfinite(lossₙ) * isfinite.(lossₙ₋ₜ) * lossₙ₋ₜ < lossₙ, lossₙ - lossₙ₋ₜ, 0.0f0)
  relΔloss = rel_stable_diff.(lossₙ, lossₙ₋ₜ)
  absrelloss = ifelse.(relΔloss .> 0.5, relΔloss, 0.5)
	@. opt_lr *= lr_rate_update_fn(absrelloss, o.opt_Δloss_up[slice])# *0.989971
	@. o.opt_lr[slice] = min(2^10, opt_lr)
end

"""
    Momentum(η = 0.01, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect damping oscillations.

# Examples
```julia
opt = Momentum()

opt = Momentum(0.01, 0.99)
```
"""
mutable struct Momentum <: AbstractOptimiser
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, IdDict())

function apply!(o::Momentum, x, Δ, idxs)
  η, ρ = o.eta, o.rho
  v = get!(() -> zero(x), o.velocity, idxs)::typeof(x)
  @. v = ρ * v - η * Δ
  @. Δ = -v
end

"""
    Nesterov(η = 0.001, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
                           prominent direction, in effect damping oscillations.

# Examples
```julia
opt = Nesterov()

opt = Nesterov(0.003, 0.95)
```
"""
mutable struct Nesterov <: AbstractOptimiser
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, IdDict())

function apply!(o::Nesterov, x, Δ, idxs)
  η, ρ = o.eta, o.rho
  v = get!(() -> zero(x), o.velocity, idxs)::typeof(x)
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  @. Δ = -d
end

"""
    RMSProp(η = 0.001, ρ = 0.9, ϵ = $ϵ64)

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
                  prominent direction, in effect damping oscillations.

# Examples
```julia
opt = RMSProp()

opt = RMSProp(0.002, 0.95)
```
"""
mutable struct RMSProp <: AbstractOptimiser
  eta::Float64
  rho::Float64
  epsilon::Float64
  acc::IdDict
end
RMSProp(η::Real = 0.001, ρ::Real = 0.9, ϵ::Real = ϵ64) = RMSProp(η, ρ, ϵ, IdDict())
RMSProp(η::Real, ρ::Real, acc::IdDict) = RMSProp(η, ρ, ϵ64, acc)

function apply!(o::RMSProp, x, Δ, idxs)
  η, ρ = o.eta, o.rho
  acc = get!(() -> zero(x), o.acc, idxs)::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ * conj(Δ)
  @. Δ *= η / (√acc + o.epsilon)
end

"""
    Adam(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

[Adam](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = Adam()

opt = Adam(0.001, (0.9, 0.8))
```
"""
mutable struct Adam <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
Adam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = ϵ64) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, ϵ64, state)

function apply!(o::Adam, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, idxs) do
      (zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
  βp .= βp .* β

  return Δ
end

"""
    RAdam(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

[Rectified Adam](https://arxiv.org/abs/1908.03265) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = RAdam()

opt = RAdam(0.001, (0.9, 0.8))
```
"""
mutable struct RAdam <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
RAdam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = ϵ64) = RAdam(η, β, ϵ, IdDict())
RAdam(η::Real, β::Tuple, state::IdDict) = RAdam(η, β, ϵ64, state)

function apply!(o::RAdam, x, Δ, idxs)
  η, β = o.eta, o.beta
  ρ∞ = 2/(1-β[2])-1

  mt, vt, βp, t = get!(o.state, idxs) do
      (zero(x), zero(x), Float64[β[1], β[2]], Ref(1))
  end :: Tuple{typeof(x),typeof(x),Vector{Float64},Base.RefValue{Int}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  ρ = ρ∞ - 2t[] * βp[2] / (1 - βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η * r
  else
    @. Δ =  mt / (1 - βp[1]) * η
  end
  βp .= βp .* β
  t[] += 1

  return Δ
end

"""
    AdaMax(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

[AdaMax](https://arxiv.org/abs/1412.6980) is a variant of Adam based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AdaMax()

opt = AdaMax(0.001, (0.9, 0.995))
```
"""
mutable struct AdaMax <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
AdaMax(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = ϵ64) = AdaMax(η, β, ϵ, IdDict())
AdaMax(η::Real, β::Tuple, state::IdDict) = AdaMax(η, β, ϵ64, state)

function apply!(o::AdaMax, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, ut, βp = get!(o.state, idxs) do
      (zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  @. Δ = (η/(1 - βp[1])) * mt/(ut + o.epsilon)
  βp .= βp .* β

  return Δ
end

"""
    OAdam(η = 0.0001, β::Tuple = (0.5, 0.9), ϵ = $ϵ64)

[OAdam](https://arxiv.org/abs/1711.00141) (Optimistic Adam)
is a variant of Adam adding an "optimistic" term suitable for adversarial training.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = OAdam()

opt = OAdam(0.001, (0.9, 0.995))
```
"""
mutable struct OAdam <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
OAdam(η::Real = 0.001, β::Tuple = (0.5, 0.9), ϵ::Real = ϵ64) = OAdam(η, β, ϵ, IdDict())
OAdam(η::Real, β::Tuple, state::IdDict) = RMSProp(η, β, ϵ64, state)

function apply!(o::OAdam, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, vt, Δ_, βp = get!(o.state, idxs) do
      (zero(x), zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),typeof(x),Vector{Float64}}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ = -Δ_
  @. Δ_ = η * mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon)
  @. Δ += 2Δ_
  βp .= βp .* β

  return Δ
end

"""
    AdaGrad(η = 0.1, ϵ = $ϵ64)

[AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.

# Examples
```julia
opt = AdaGrad()

opt = AdaGrad(0.001)
```
"""
mutable struct AdaGrad <: AbstractOptimiser
  eta::Float64
  epsilon::Float64
  acc::IdDict
end
AdaGrad(η::Real = 0.1, ϵ::Real = ϵ64) = AdaGrad(η, ϵ, IdDict())
AdaGrad(η::Real, state::IdDict) = AdaGrad(η, ϵ64, state)

function apply!(o::AdaGrad, x, Δ, idxs)
  η = o.eta
  acc = get!(() -> fill!(similar(x), o.epsilon), o.acc, idxs)::typeof(x)
  @. acc += Δ * conj(Δ)
  @. Δ *= η / (√acc + o.epsilon)
end

"""
    AdaDelta(ρ = 0.9, ϵ = $ϵ64)

[AdaDelta](https://arxiv.org/abs/1212.5701) is a version of AdaGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.

# Examples
```julia
opt = AdaDelta()

opt = AdaDelta(0.89)
```
"""
mutable struct AdaDelta <: AbstractOptimiser
  rho::Float64
  epsilon::Float64
  state::IdDict{Any, Any}
end
AdaDelta(ρ::Real = 0.9, ϵ::Real = ϵ64) = AdaDelta(ρ, ϵ, IdDict())
AdaDelta(ρ::Real, state::IdDict) = AdaDelta(ρ, ϵ64, state)

function apply!(o::AdaDelta, x, Δ, idxs)
  ρ = o.rho
  acc, Δacc = get!(() -> (zero(x), zero(x)), o.state, idxs)::NTuple{2,typeof(x)}
  @. acc = ρ * acc + (1 - ρ) * Δ * conj(Δ)
  # DON'T remove epsilon from numerator
  # or even out of the square roots
  @. Δ *= √(Δacc + o.epsilon) / √(acc + o.epsilon)
  @. Δacc = ρ * Δacc + (1 - ρ) * Δ * conj(Δ)
  return Δ
end

"""
    AMSGrad(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the Adam
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AMSGrad()

opt = AMSGrad(0.001, (0.89, 0.995))
```
"""
mutable struct AMSGrad <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64, Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
AMSGrad(η::Real = 0.001, β = (0.9, 0.999), ϵ::Real = ϵ64) = AMSGrad(η, β, ϵ, IdDict())
AMSGrad(η::Real, β::Tuple, state::IdDict) = AMSGrad(η, β, ϵ64, state)

function apply!(o::AMSGrad, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, vt, v̂t = get!(o.state, idxs) do
    (fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon), fill!(similar(x), o.epsilon))
  end :: NTuple{3,typeof(x)}

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  @. Δ = η * mt / (√v̂t + o.epsilon)
end

"""
    NAdam(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

[NAdam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) is a Nesterov variant of Adam.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = NAdam()

opt = NAdam(0.002, (0.89, 0.995))
```
"""
mutable struct NAdam <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64, Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
NAdam(η::Real = 0.001, β = (0.9, 0.999), ϵ::Real = ϵ64) = NAdam(η, β, ϵ, IdDict())
NAdam(η::Real, β::Tuple, state::IdDict) = NAdam(η, β, ϵ64, state)

function apply!(o::NAdam, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, idxs) do
    (zero(x), zero(x), Float64[o.beta[1], o.beta[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float64}}
  β1p, β2p = βp

  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
  @. Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + o.epsilon) * η
  βp .= βp .* β

  return Δ
end

"""
    AdamW(η = 0.001, β::Tuple = (0.9, 0.999), decay = 0)

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- `decay`: Decay applied to weights during optimisation.

# Examples
```julia
opt = AdamW()

opt = AdamW(0.001, (0.89, 0.995), 0.1)
```
"""
AdamW(η = 0.001, β = (0.9, 0.999), decay = 0) =
  Optimiser(Adam(η, β), WeightDecay(decay))

"""
    AdaBelief(η = 0.001, β::Tuple = (0.9, 0.999), ϵ = $ϵ64)

The [AdaBelief](https://arxiv.org/abs/2010.07468) optimiser is a variant of the well-known
Adam optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.

# Examples
```julia
opt = AdaBelief()

opt = AdaBelief(0.001, (0.9, 0.8))
```
"""
mutable struct AdaBelief <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  epsilon::Float64
  state::IdDict{Any, Any}
end
AdaBelief(η::Real = 0.001, β = (0.9, 0.999), ϵ::Real = ϵ64) = AdaBelief(η, β, ϵ, IdDict())
AdaBelief(η::Real, β::Tuple, state::IdDict) = AdaBelief(η, β, ϵ64, state)

function apply!(o::AdaBelief, x, Δ, idxs)
  η, β = o.eta, o.beta

  mt, st, βp = get!(o.state, idxs) do
      (zero(x), zero(x), Float64[β[1], β[2]])
  end :: Tuple{typeof(x), typeof(x), Vector{Float64}}

  #= st is a variance and can go to zero. This is in contrast to Adam, which uses the
  second moment which is usually far enough from zero. This is problematic, since st
  can be slightly negative due to numerical error, and the square root below will fail.
  Also, if we want to differentiate through the optimizer, √0 is not differentiable.
  To protect against this, we add a small number, st -> st + eps2.
  The original implementation (https://github.com/juntang-zhuang/Adabelief-Optimizer)
  uses the square of Adam's epsilon, which we do here.
  See also: https://github.com/juntang-zhuang/Adabelief-Optimizer/issues/61 =#
  eps2 = o.epsilon^2 # TODO: make epsilon^2 the default in next breaking release
  
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. st = β[2] * st + (1 - β[2]) * (Δ - mt) * conj(Δ - mt) + eps2
  @. Δ =  η * mt / (1 - βp[1]) / (√(st / (1 - βp[2])) + eps2)
  βp .= βp .* β

  return Δ
end


# Compose optimizers

"""
    Optimiser(a, b, c...)

Combine several optimisers into one; each optimiser produces a modified gradient
that will be fed into the next, and this is finally applied to the parameter as
usual.
"""
mutable struct Optimiser <: AbstractOptimiser
  os::Vector{Any}
end

Optimiser(opts::AbstractOptimiser...) = Optimiser(Any[opts...])


Base.getindex(c::Optimiser, i::AbstractArray) = Optimiser(c.os[i]...)

function apply!(o::Optimiser, x, Δ)
  for opt in o.os
    Δ = apply!(opt, x, Δ)
  end
  return Δ
end

"""
    InvDecay(γ = 0.001)

Apply inverse time decay to an optimiser, so that the effective step size at
iteration `n` is `eta / (1 + γ * n)` where `eta` is the initial step size.
The wrapped optimiser's step size is not modified.

See also the [Scheduling Optimisers](@ref) section of the docs
for more general scheduling techniques.

# Examples

`InvDecay` is typically composed  with other optimizers 
as the last transformation of the gradient:

```julia
# Inverse decay of the learning rate
# with starting value 0.001 and decay coefficient 0.01.
opt = Optimiser(Adam(1f-3), InvDecay(1f-2))
```
"""
mutable struct InvDecay <: AbstractOptimiser
  gamma::Float64
  state::IdDict{Any, Int}
end

InvDecay(γ = 0.001) = InvDecay(γ, IdDict{Any, Int}())

function apply!(o::InvDecay, x, Δ, idxs)
  γ = o.gamma
  n = get!(o.state, idxs, 1)
  Δ .*= 1 / (1 + γ * n)
  o.state[x] = n + 1
  return Δ
end

"""
    ExpDecay(η = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, start = 1)

Discount the learning rate `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- `decay`: Factor by which the learning rate is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of learning rate.
- 'start': Step at which the decay starts.


See also the [Scheduling Optimisers](@ref) section of the docs
for more general scheduling techniques.

# Examples

`ExpDecay` is typically composed  with other optimizers 
as the last transformation of the gradient:
```julia
opt = Optimiser(Adam(), ExpDecay(1.0))
```
Note: you may want to start with `η=1` in `ExpDecay` when combined with other
optimizers (`Adam` in this case) that have their own learning rate.
"""
mutable struct ExpDecay <: AbstractOptimiser
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
  start::Int64
  current::IdDict
end

ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, start = 0) =
  ExpDecay(opt, decay, decay_step, clip, start, IdDict())

function apply!(o::ExpDecay, x, Δ, idxs)
  η, s, decay, start = o.eta, o.step, o.decay, o.start
  n = o.current[x] = get(o.current, idxs, 0) + 1
  if n > start && n % s == 0 && count(x -> x > start && x % s == 0, values(o.current)) == 1
    η = max(η * decay, o.clip)
    o.eta = η
  end
  @. Δ *= η
end

"""
    WeightDecay(λ = 0)

Decay weights by ``λ``. 
Typically composed  with other optimizers as the first transformation to the gradient,
making it equivalent to adding ``L_2`` regularization 
with coefficient  ``λ`` to the loss.

# Examples

```julia
opt = Optimiser(WeightDecay(1f-4), Adam())
```
"""
mutable struct WeightDecay <: AbstractOptimiser
  wd::Real
end

WeightDecay() = WeightDecay(0)

function apply!(o::WeightDecay, x, Δ, idxs)
  wd = o.wd
  @. Δ += wd * x
end

"""
    ClipValue(thresh)

Clip gradients when their absolute value exceeds `thresh`.
"""
mutable struct ClipValue{T} <: AbstractOptimiser
    thresh::T
end

apply!(o::ClipValue, x, Δ, idxs) = clamp!(Δ, -o.thresh, o.thresh)

"""
    ClipNorm(thresh)

Clip gradients when their L2 norm exceeds `thresh`.
"""
mutable struct ClipNorm{T} <: AbstractOptimiser
    thresh::T
end

function apply!(o::ClipNorm, x, Δ, idxs)
    Δnrm = norm(Δ)
    if Δnrm > o.thresh
        rmul!(Δ, o.thresh / Δnrm)
    end
    return Δ
end
end # module Optimizers

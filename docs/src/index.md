## Beta-scaling equation

The beta-scaling model is implemented to make it easier to find critical exponents of MCT. The equation is

$$\sigma - \delta t + \lambda (g(t))^2 = \partial_t∫g(t-\tau)g(\tau)d\tau.$$

Here, $\sigma$ is the distance from the critical point, $\lambda$ is the relevant eigenvalue of the stability matrix. $g(t)$ describes the deviation of the order parameter from the plateau. $\delta$ is an optional hopping parameter, defaulting to 0 if not specified. Each of the parameters have to be floating point numbers.  


### Example
In order to solve the beta-scaling equation, we have to specify the parameters defining the equation and a time-scale `t0` that shifts the results. 
```julia
using ModeCouplingTheory
using MCTBetaScaling, Plots
λ = 0.7; ϵ = -0.1; t0 = 0.001
equation = BetaScalingEquation(λ, ϵ, t0)
sol = solve(equation, TimeDoublingSolver(t_max=10^4.))
plot(log10.(sol.t), log10.(abs.(sol.F)), ylabel="log_{10}(|g(t)|)", xlabel="log_{10}(t)", label="g(t)")
```

![image](images/beta.png)

In the figure, the slopes of the straight lines are given by the parameters $-a$ and $b$, which describe the relaxation towards and away from the plateau value of the correlator. These exponents are automatically computed, and are stored in `equation.coeff.a` and `equation.coeff.b`.

## References
Götze, J Phys Condens Matter 2, 8485 (1990)


## Stochastic Beta-Relaxation (SBR)

SBR is an extension of the beta-scaling equation, where the parameter $\sigma$ becomes quenched disorder, and a diffusive term is added
$$\sigma(x) + \alpha \nabla^2g(x,t) - \delta t + \lambda (g(x,t))^2 = \partial_t\intg(x,t-\tau)g(x,\tau)d\tau.$$

This is implemented in 1, 2, and 3 dimensions with periodic boundaries. Example:

```julia
solver = TimeDoublingSolver(t_max=10^10., verbose=true, tolerance=1e-10, N=64, Δt=1e-12)

L_sys = 100.0 ## physical size of the system
n = 100 ## number of sites on one side of the lattice
dims = 2 
Ls = ntuple(i -> L_sys,  dims)  # Lattice size in each dimension
ns = ntuple(i -> n, dims)  # Number of sites in one dimension of the lattice

λ = 0.75
α = 0.1
t₀ = 0.001
sigma2 = 0.01  # desired variance
sigma_limits = sqrt(3 * sigma2)
x(ϵ) = rand() * 2 * ϵ - ϵ  # Uniform in (-ε, +ε)
σ_vec = [x(sigma_limits) for i in 1:prod(ns)]  # small random variations near σ = 0

eqn_sys = MCTBetaScaling.StochasticBetaScalingEquation(λ, α, σ_vec, t₀, Ls, ns)
sol = solve(eqn_sys, solver)

```
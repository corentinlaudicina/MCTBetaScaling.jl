import Pkg;
Pkg.activate(".")

using ModeCouplingTheory, CairoMakie
include("StochasticBetaScaling.jl")


Nx = 10
σ_vec = randn(Nx) * 0.01   # small random variations near σ = 0
λ = 0.75
t₀ = 0.001
eqn_sys = StochasticBetaScalingEquation(λ, σ_vec, t₀)

sol = solve(eqn_sys, TimeDoublingSolver(t_max=10^4.))

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1],
            title="β-scaling equation solution",
            xlabel="t",
            ylabel="|g(t)|",
            yscale=log10,
            xscale=log10)
for i in Nx
    F = get_F(sol, :, i)
    lines!(ax, sol.t[2:end], abs.(F[2:end]), label="g(t)")
end

display(fig)

    # λ = 0.7; ϵ = -0.1; t0 = 0.001
# equation = StochasticBetaScalingEquation(λ, ϵ, t0)
# sol = solve(equation, TimeDoublingSolver(t_max=10^4.))

# fig = Figure(size=(400,400))
# ax = Axis(fig[1, 1], 
#           title="β-scaling equation solution", 
#           xlabel="t", 
#           ylabel="|g(t)|", 
#           yscale=log10, 
#           xscale=log10)

# lines!(ax, sol.t[2:end], abs.(sol.F[2:end]), label="g(t)")

# display(fig)

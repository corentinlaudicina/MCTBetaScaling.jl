import Pkg;
Pkg.activate(".")
using Revise, Random
using ModeCouplingTheory, CairoMakie
include("MCTBetaScaling.jl")

Random.seed!(1234)  # For reproducibility

solver = TimeDoublingSolver(t_max=10^10., verbose=true, tolerance=1e-10, N=1024, Δt=1e-6)

L_sys = 10.0 ## physical size of the system
L = 10 ## number of sites on one side of the square lattice
dx = L_sys / L

λ = 0.75
α = 0.01
t₀ = 0.001

sigma2 = 0.01  # desired variance
sigma_limits = sqrt(3 * sigma2)
x(ϵ) = rand() * 2 * ϵ - ϵ  # Uniform in (-ε, +ε)

σ_vec = [x(sigma_limits) for i in 1:L^2]  # small random variations near σ = 0

n_pos = sum(σ > 0 for σ in σ_vec)
n_neg = sum(σ < 0 for σ in σ_vec)

println("Number of positive σ: $n_pos, negative σ: $n_neg")

eqn_sys = MCTBetaScaling.StochasticBetaScalingEquation(λ, α, σ_vec, t₀, L_sys)
sol = @time solve(eqn_sys, solver)

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1],
            title="β-scaling equation solution",
            xlabel="t",
            ylabel="|g(t)|",
            yscale=log10,
            xscale=log10,
            limits=(1e-10, 1e18, 1e-4, 1e6),
         )

for i in 1:L^2
    F = get_F(sol, :, i)
    lines!(ax, sol.t[2:end], abs.(F[2:end]))
end

display(fig)

import Pkg;
Pkg.activate(".")
using Revise, Random
using ModeCouplingTheory, CairoMakie
include("../src/MCTBetaScaling.jl")

Random.seed!(1234)  # For reproducibility

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

n_pos = sum(σ > 0 for σ in σ_vec)
n_neg = sum(σ < 0 for σ in σ_vec)

println("Number of positive σ: $n_pos, negative σ: $n_neg")

eqn_sys = MCTBetaScaling.StochasticBetaScalingEquation(λ, α, σ_vec, t₀, Ls, ns)
sol = @time solve(eqn_sys, solver)
fig = Figure(size=(1400, 1400))
ax = Axis(fig[1, 1],
            title="β-scaling equation solution",
            xlabel="t",
            ylabel="|g(t)|",
            yscale=log10,
            xscale=log10,
            limits=(1e-10, 1e6, 1e-4, 1e6),
         )

for i in 1:ns[1]:prod(ns)
    F = get_F(sol, :, i)
    scatterlines!(ax, sol.t, abs.(F))
end

display(fig)

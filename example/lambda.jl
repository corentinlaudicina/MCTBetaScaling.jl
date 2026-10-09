#
# Compute the MCT exponent parameter λ for Percus–Yevick hard spheres
# and feed it into the β-scaling equation.
#
import Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
using LinearAlgebra, ModeCouplingTheory, MCTBetaScaling

# --- Wavevector grid and thermodynamic state -------------------------------
# η is chosen just above the PY hard-sphere MCT transition, η_c ≈ 0.51593, so
# that the largest eigenvalue of the stability matrix sits at ≈ 1 and λ is the
# universal exponent parameter.
η   = 0.51593
ρ   = 6 / π * η
kBT = 1.0
m   = 1.0

Nk   = 100
kmax = 40.0
dk   = kmax / Nk
k    = dk .* (collect(1:Nk) .- 0.5)          # midpoint grid required by the kernel

Sk = S_PY.(k, η)                             # Percus–Yevick structure factor
γ  = @. k^2 * kBT / (m * Sk)                  # bare relaxation rate D₀ q² / S(q)

# --- Critical nonergodicity parameter -------------------------------------
kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k, Sk, 3)
sol    = solve_steady_state(γ, Sk, kernel; tolerance = 1e-8, verbose = false)
fc     = get_F(sol, 1, :) ./ Sk              # normalized f_c(k) = F_c(k) / S(k)

# --- Exponent parameter --------------------------------------------------
λ, hR, hL, C1 = compute_lambda(kernel, fc)

E0 = maximum(real.(eigvals(C1)))
println("largest eigenvalue of the stability matrix : ", round(E0; digits = 5))
println("exponent parameter λ                       : ", round(λ; digits = 6))

# --- Critical exponents and a β-scaling solution -------------------------
# BetaScalingEquation derives the exponents a, b from λ via
#   λ = Γ(1-a)²/Γ(1-2a) = Γ(1+b)²/Γ(1+2b).
ϵ   = -1e-4                                   # slightly on the liquid side
t₀  = 1e-4
eqn = BetaScalingEquation(λ, ϵ, t₀)
βsol = solve(eqn, TimeDoublingSolver(t_max = 1e10))

a = eqn.coeffs.a
b = eqn.coeffs.b
println("critical decay exponent  a                  : ", round(a; digits = 5))
println("von Schweidler exponent  b                  : ", round(b; digits = 5))
println("relaxation-time exponent γ = 1/2a + 1/2b    : ", round(1 / (2a) + 1 / (2b); digits = 5))

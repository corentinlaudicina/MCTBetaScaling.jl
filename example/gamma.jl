#
# Coefficient Γ of the gradient term of the stochastic β-scaling equation,
#
#   σ(x) - δ t + λ g(x,t)² + Γ ∇²g(x,t) = ∂ₜ∫ g(x,t-τ) g(x,τ) dτ ,
#
# i.e. the q² coefficient of the top eigenvalue of the MCT stability operator, E0(q) = E0 - Γ q² + O(q⁴), for Percus–Yevick hard spheres.

import Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
using MCTBetaScaling

# --- Wavevector grid -------------------------------------------------------
Nk   = 200
kmax = 60.0
dk   = kmax / Nk
k    = dk .* (collect(1:Nk) .- 0.5)

# --- The system as a function of the packing fraction ----------------------
# Γ needs c′(k) and c″(k). For PY they are exact (dcf_PY); for a tabulated c(k),leave out `dc` and compute_gamma takes them from a quintic spline (dcf_derivatives). The bipolar weights depend on the grid only, so every φ shares the template's.

template = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), 3)
function system(φ)
    c, c1, c2 = dcf_PY(k, φ)
    S = @. 1 / (1 - 6φ / π * c)
    return BipolarMCTKernel(template, 6φ / π, S), (c1, c2)
end

# --- Critical point of this grid ---------------------------------------------
# Γ, like λ, drifts like √ε away from φc, so locate φc of the discrete equation first.
crit = find_critical_packing(system, 0.52)
φc   = crit.φc
println("critical packing fraction φc               : ", round(φc; digits=10))

# --- One glass state close to it -----------------------------------------------
ε = 1e-5
K, dc = system(φc * (1 + ε))
F = steady_state(K)                           # plateau F∞ = S f (unnormalised)
G = compute_gamma(K, F ./ K.Sk; dc)           # also accepts a dDimModeCouplingKernel
println("\nat ε = $ε:")
println("  top eigenvalue E0                        : ", round(G.E0; digits=6))
println("  Γ = Γ_A + Γ_V                            : ", round(G.Γ; sigdigits=5),
        " = ", round(G.Γ_A; sigdigits=5), " + ", round(G.Γ_V; sigdigits=5))
println("  correlation length² ξ² = Γ/(1 - E0)      : ", round(G.Γ / (1 - G.E0); sigdigits=5))

# --- Critical values, ε → 0 ----------------------------------------------------
# glass_sweep solves the plateau from the largest ε down by continuation;
# sweep_summary fits each quantity as X = Xc + a√ε + bε + cε^{3/2}.
rows = glass_sweep(system, φc, [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
sm   = sweep_summary(rows)
println("\nε → 0:")
println("  Γc                                       : ", round(sm.Γc; sigdigits=5))
println("  λc                                       : ", round(sm.λc; sigdigits=5))
println("  2g(1-λ), from 1 - E0 = 2g(1-λ)√ε         : ", round(sm.two_g_one_minus_λ; sigdigits=5))
# ξ₀² is what Biroli et al.'s (iMCT PRL, (2006)) fit λ_q/λ_0 - 1 = 0.072 q² ε^{-1/2} measures.
println("  ξ₀² = Γc / 2g(1-λ)  (ξ = ξ₀ ε^{-1/4})    : ", round(sm.ξ0sq; sigdigits=5))

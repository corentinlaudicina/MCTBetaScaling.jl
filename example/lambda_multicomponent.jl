#
# N=1 reduction demo:  compute_lambda_multicomponent  vs  compute_lambda
# for Percus–Yevick hard spheres just above the MCT transition.
#
import Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
using LinearAlgebra, ModeCouplingTheory, MCTBetaScaling
import ModeCouplingTheory: dDimModeCouplingKernel, MultiComponentModeCouplingKernel
const SMatrix = ModeCouplingTheory.SMatrix

η = 0.51593; ρ = 6 / π * η; kBT = 1.0; m = 1.0
Nk = 100; kmax = 40.0; dk = kmax / Nk
k = dk .* (collect(1:Nk) .- 0.5)
Sk = S_PY.(k, η)
γ  = @. k^2 * kBT / (m * Sk)

# --- single component ---
kernel = dDimModeCouplingKernel(ρ, kBT, m, k, Sk, 3)
sol    = solve_steady_state(γ, Sk, kernel; tolerance = 1e-8, verbose = false)
fc     = get_F(sol, 1, :) ./ Sk
λ, hR, hL, C1 = compute_lambda(kernel, fc)

# --- trivial 1-species mixture, same S(k) ---
kernel_mc = MultiComponentModeCouplingKernel([ρ], kBT, [m], k, [SMatrix{1,1,Float64}(Sk[i]) for i in 1:Nk])
f_mc      = [SMatrix{1,1,Float64}(Sk[i] * fc[i]) for i in 1:Nk]          # un-normalized NEP  F_cᵅᵝ(k)
λm, hRm, hLm, C1m = compute_lambda_multicomponent(kernel_mc, f_mc)

println("compute_lambda               λ = ", λ)
println("compute_lambda_multicomponent λ = ", λm)
println("relative difference             = ", abs(λm - λ) / abs(λ))
println("E0 (scalar / matrix)            = ", maximum(real.(eigvals(C1))), " / ", maximum(real.(eigvals(C1m))))

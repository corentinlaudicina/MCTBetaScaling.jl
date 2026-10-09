using LinearAlgebra
import ModeCouplingTheory: dDimModeCouplingKernel, MultiComponentModeCouplingKernel, evaluate_kernel
const SMatrix = ModeCouplingTheory.SMatrix

# ---------------------------------------------------------------------------
# N = 1 reduction check:  compute_lambda_multicomponent must reproduce
# compute_lambda for a trivial 1-species mixture built from the same S(k),
# f_c(k) as the single-component Percus–Yevick hard-sphere test.
# ---------------------------------------------------------------------------
@testset "lambda_multicomponent N=1 reduction" begin

    η = 0.51593
    ρ = η * 6 / π; kBT = 1.0; m = 1.0
    Nk = 100; kmax = 40.0
    dk = kmax / Nk
    k_array = dk * (collect(1:Nk) .- 0.5)
    Sₖ = S_PY.(k_array, η)
    γ = @. k_array^2 * kBT / (m * Sₖ)

    # --- single-component reference ---
    kernel_sc = dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)
    sol = solve_steady_state(γ, Sₖ, kernel_sc; tolerance = 1e-8, verbose = false)
    fk = get_F(sol, 1, :) ./ Sₖ                     # normalized f_c(k)
    λ_sc, hR_sc, hL_sc, C1_sc = compute_lambda(kernel_sc, fk)
    E0_sc = maximum(real.(eigvals(C1_sc)))

    # --- trivial 1-species MultiComponentModeCouplingKernel from the same S(k) ---
    Sk_mc = [SMatrix{1,1,Float64}(Sₖ[i]) for i in 1:Nk]
    kernel_mc = MultiComponentModeCouplingKernel([ρ], kBT, [m], k_array, Sk_mc)

    # un-normalized NEP matrix field  F_cᵅᵝ(k) = f_c(k) S(k)
    f_mc = [SMatrix{1,1,Float64}(Sₖ[i] * fk[i]) for i in 1:Nk]
    λ_mc, hR_mc, hL_mc, C1_mc = compute_lambda_multicomponent(kernel_mc, f_mc)
    E0_mc = maximum(real.(eigvals(C1_mc)))

    # the actual N=1 reduction check: multi-component result equals the scalar one
    @test isapprox(λ_mc, λ_sc; rtol = 1e-8)
    @test isapprox(E0_mc, E0_sc; atol = 1e-8)
    @test maximum(real.(eigvals(C1_mc))) ≈ 1.0 atol = 1e-2      # close to criticality
    # same universal value the scalar test pins (looser rtol: solver tolerance only)
    @test λ_mc ≈ 0.7222348083592552 rtol = 1e-5

    # hR agrees with the scalar right eigenvector up to the normalization
    # convention:  the matrix operator acts on the un-normalized field, so
    # hR_mc(k) ∝ S(k) * hR_sc(k).
    r = [hR_mc[i][1, 1] / (Sₖ[i] * hR_sc[i]) for i in 1:Nk]
    @test maximum(abs, r .- sum(r) / Nk) / abs(sum(r) / Nk) < 1e-6

    # λ and E0 are dimensionless and the plateau does not depend on kBT or m, so
    # rescaling them must change nothing (the λ term used to miss the W/kBT weight).
    kernel_mc2 = MultiComponentModeCouplingKernel([ρ], 2.0, [3.0], k_array, Sk_mc)
    λ_mc2, _, _, C1_mc2 = compute_lambda_multicomponent(kernel_mc2, f_mc)
    @test isapprox(λ_mc2, λ_mc; rtol = 1e-10)
    @test isapprox(maximum(real.(eigvals(C1_mc2))), E0_mc; atol = 1e-10)

    # Nk×Ns×Ns array input is accepted and gives the same λ
    f_arr = zeros(Nk, 1, 1); for i in 1:Nk; f_arr[i, 1, 1] = Sₖ[i] * fk[i]; end
    λ_arr, hR_arr, _, _ = compute_lambda_multicomponent(kernel_mc, f_arr)
    @test isapprox(λ_arr, λ_sc; rtol = 1e-8)
    @test size(hR_arr) == (Nk, 1, 1)
end

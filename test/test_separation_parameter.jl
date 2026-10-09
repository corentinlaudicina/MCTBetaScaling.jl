using LinearAlgebra

@testset "separation parameter" begin

    # Same PY hard-sphere critical point as the λ test, η_c ≈ 0.51593.
    η = 0.51593;
    ρ = η*6/π; kBT = 1.0; m = 1.0

    Nk = 100; kmax = 40.0
    dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5)
    Sₖ = S_PY.(k_array, η)
    γ = @. k_array^2*kBT/(m*Sₖ)
    kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)
    sol = solve_steady_state(γ, Sₖ, kernel; tolerance=10^-8, verbose=false)
    fk = get_F(sol, 1, :) ./ Sₖ

    # dc/dn = (π/6) dc/dη, central difference on the closed-form PY c(k).
    h = 1e-6
    dcdn = (c_PY.(k_array, η + h) .- c_PY.(k_array, η - h)) ./ (2h) .* (π/6)

    λ, hR, hL, C1 = compute_lambda(kernel, fk)
    C, C0 = compute_C(kernel, fk, dcdn, hL)
    @test compute_C(kernel, fk, dcdn)[1] ≈ C

    # --- C⁽⁰⁾ is (1-f_c)² n ∂𝓕[f_c]/∂n at fixed normalized f_c --------------------
    # Rebuild the kernel at η ± δ and difference the memory functional directly; this
    # pins the Ornstein–Zernike dS/dn, the cross vertex and the leg term at once.
    function memory(η′)
        S′ = S_PY.(k_array, η′)
        kk = ModeCouplingTheory.dDimModeCouplingKernel(η′*6/π, kBT, m, k_array, S′, 3)
        F = fk .* S′
        out = zeros(Nk)
        for q in 1:Nk, j in 1:Nk, l in abs(q-j)+1 : min(Nk, q+j-1)
            out[q] += kk.prefactor * kk.J[q, j, l] * kk.V[q, j, l] * F[j] * F[l]
        end
        return out .* S′ ./ (kBT/m .* k_array.^2)
    end
    δ = 1e-5
    n_dFdn = η .* (memory(η + δ) .- memory(η - δ)) ./ (2δ)        # n ∂/∂n = η ∂/∂η
    @test maximum(abs, C0 .- (1 .- fk).^2 .* n_dFdn) < 1e-6 * maximum(abs, C0)

    # --- literature -------------------------------------------------------------
    @test C ≈ 1.5520870844912167 rtol=1e-6    # regression pin, Nk=100, kmax=40
    # σ = C ε with C ≈ 1.54 for PY hard spheres (Franosch et al. 1997, λ = 0.735
    # there vs 0.722 here, which is a different discretization of the vertex).
    @test isapprox(C, 1.54, rtol=0.02)

end

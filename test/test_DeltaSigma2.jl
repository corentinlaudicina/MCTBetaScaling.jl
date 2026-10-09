using LinearAlgebra

@testset "DeltaSigma2" begin

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

    λ, hR, hL, C1 = compute_lambda(kernel, fk)
    μ = compute_mu(kernel, fk, hL)
    @test compute_mu(kernel, fk) ≈ μ

    # the kernel's c(k) is the PY direct correlation function the grid was built from
    @test kernel.C ≈ c_PY.(k_array, η)

    @test μ ≈ -0.012334664133691468 rtol=1e-6     # regression pin, Nk=100, kmax=40

    # --- S(0) -------------------------------------------------------------------
    # extrapolating c(k) in k² from the grid must reproduce the exact PY S(0)
    @test S0_PY(η) ≈ (1 - η)^4 / (1 + 2η)^2
    @test extrapolate_S0(kernel) ≈ S0_PY(η) rtol=1e-5

    # --- Δσ² = μ² n_c S(0) --------------------------------------------------------
    Δσ² = compute_DeltaSigma2(kernel, fk, hL; S0=S0_PY(η))
    @test Δσ² ≈ μ^2 * ρ * S0_PY(η)
    @test compute_DeltaSigma2(kernel, fk; S0=S0_PY(η)) ≈ Δσ²
    @test compute_DeltaSigma2(kernel, fk, hL) ≈ Δσ² rtol=1e-5     # extrapolated S(0)

end

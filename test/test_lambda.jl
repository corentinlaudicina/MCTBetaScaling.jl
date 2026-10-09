using LinearAlgebra

@testset "lambda" begin

    # We solve MCT for hard spheres close to the PY glass transition, η_c ≈ 0.51593

    η = 0.51593;
    ρ = η*6/π; kBT = 1.0; m = 1.0

    Nk = 100; kmax = 40.0;
    dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5) # construct the grid this way to satisfy the assumptions
                                                    # of the discretization.
    Sₖ = S_PY.(k_array, η)

    γ = @. k_array^2*kBT/(m*Sₖ)   # bare relaxation rate D₀ q² / S(q)

    kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)
    sol = solve_steady_state(γ, Sₖ, kernel; tolerance=10^-8, verbose=false)

    # compute_lambda takes the kernel and the *normalized* critical nonergodicity
    # parameter f_c(k) = F_c(k) / S(k) ∈ [0, 1].
    fk = get_F(sol, 1, :) ./ Sₖ

    λ, hR, hL, C1 = compute_lambda(kernel, fk)
    @test maximum(real.(eigvals(C1))) ≈ 1.0 atol=1e-2 # close to the critical point
    @test λ ≈ 0.7222348083592552

end
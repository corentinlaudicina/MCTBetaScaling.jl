

@testset "lambda" begin

    function find_analytical_C_k(k, η)
        A = -(1 - η)^-4 *(1 + 2η)^2
        B = (1 - η)^-4*  6η*(1 + η/2)^2
        D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
        Cₖ = @. 4π/k^6 * 
        (
            24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k)
        + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k)
        )
        return Cₖ
    end

    function find_analytical_S_k(k, η)
            Cₖ = find_analytical_C_k(k, η)
            ρ = 6/π * η
            Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
        return Sₖ
    end

    # We solve MCT for hard spheres at a volume fraction of 0.51591

    η = 0.54; 
    ρ = η*6/π; kBT = 1.0; m = 1.0

    Nk = 40; kmax = 40.0; 
    dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5) # construct the grid this way to satisfy the assumptions
                                                    # of the discretization.
    Sₖ = find_analytical_S_k(k_array, η)

    ∂F0 = zeros(Nk); α = 1.0; β = 0.0; γ = @. k_array^2*kBT/(m*Sₖ); δ = 0.0

    
    kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)
    sol = solve_steady_state(γ, Sₖ, kernel; tolerance=10^-8, verbose=false)
    fk = get_F(sol, 1, :)

    # in principle the vertex should be constructre by the user.
    # here, the implemenation is 
    # K[q] = kernel.prefactor[q] * sum_kp kernel.V[q,k,p] * kernel.J[q,k,p] * F[k] * F[p]
    # So we can construct the vertex directly from the kernel as follows:
    V = kernel.prefactor .* kernel.V .* kernel.J

    # Ensure that the vertex is correctly normalized
    for q in 1:Nk, k in 1:Nk, p in 1:Nk
        V[q,k,p] = V[q,k,p] * Sₖ[k] * Sₖ[p] / γ[q]
    end

    # normalize fk
    fk = fk ./ Sₖ

    λ, r, l, C = compute_lambda(V, fk; check_symmetry=true)
    @test λ ≈ 0.4118771387551784

end
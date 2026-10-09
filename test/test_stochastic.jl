# End-to-end checks of StochasticBetaScalingEquation against oracles that share none of
# its time-stepping code: the scalar β equation (α = 0 decouples the sites), the exact
# glass plateau, and the vanishing Laplacian of a uniform field.

using Test, ModeCouplingTheory, MCTBetaScaling

β_solver() = TimeDoublingSolver(t_max=1e6, Δt=1e-8, N=64)

@testset "α = 0: each site is the scalar β solution for its own σ, δ" begin
    λ, t0 = 0.75, 1e-3
    σs = [-0.1, -0.01, 0.05, 0.1]                # liquid, liquid, glass, glass + hopping
    δs = [0.0, 0.0, 0.0, 1e-3]
    sol = solve(StochasticBetaScalingEquation(λ, 0.0, σs, t0, (2.0, 2.0), (2, 2); δ=δs), β_solver())
    G = reduce(hcat, sol.F)                      # sites × times
    late = lastindex(sol.t)-199:lastindex(sol.t)
    for i in eachindex(σs)
        ref = solve(BetaScalingEquation(λ, σs[i], t0; δ=δs[i]), β_solver())
        @test ref.t ≈ sol.t
        # The lattice code also starts from the A1·σ·tᵃ correction, so early times differ
        # slightly by design; late times must agree (measured ≤ 8.5e-5). Counting the
        # middle time cell twice, for instance, shifts them by ~1%.
        @test maximum(abs.(G[i, late] .- ref.F[late]) ./ abs.(ref.F[late])) < 5e-4
    end
    # glass side without hopping: g(t → ∞) = √(σ/(1-λ))
    @test G[3, end] ≈ sqrt(σs[3] / (1 - λ)) rtol = 1e-8
end

@testset "uniform σ: the Laplacian term drops out" begin
    λ, t0, σ = 0.75, 1e-3, -0.02
    ns, Ls = (4, 4), (4.0, 4.0)
    free     = solve(StochasticBetaScalingEquation(λ, 0.0, fill(σ, prod(ns)), t0, Ls, ns), β_solver())
    coupled  = solve(StochasticBetaScalingEquation(λ, 0.5, fill(σ, prod(ns)), t0, Ls, ns), β_solver())
    A0, A1 = reduce(hcat, free.F), reduce(hcat, coupled.F)
    @test all(c -> all(==(c[1]), c), eachcol(A1))   # every site sees the same neighbours
    @test A1 ≈ A0 rtol = 1e-6                         # measured 2e-8
end

@testset "strong-coupling warning" begin
    # αeff = α/dx² with dx = 1
    @test_logs (:warn, r"Strong lattice coupling") StochasticBetaScalingEquation(0.75, 60.0, zeros(4), 1e-3, (2.0, 2.0), (2, 2))
    @test_logs StochasticBetaScalingEquation(0.75, 0.1, zeros(4), 1e-3, (2.0, 2.0), (2, 2))
end

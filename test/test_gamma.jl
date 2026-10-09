using LinearAlgebra, ForwardDiff, Random

@testset "gamma" begin
    MB = MCTBetaScaling
    D(f, x)  = ForwardDiff.derivative(f, x)
    D2(f, x) = D(y -> D(f, y), x)

    # Smooth stand-ins for c, A = S(1-f) and B = S f, for the algebra tests.
    c(x) = -exp(-0.3x^2) * (1 + 0.5cos(2.1x))
    A(x) = 1.3 + 0.4exp(-0.2x^2) * cos(x)
    B(x) = exp(-0.1x^2) * (1 + 0.3sin(x))
    ρ = 0.9

    # U(q) = ṽ_{k+q/2}(k-p, p+q/2) and the full operator C_q(k,p), no expansion.
    U(kv, pv, q) = (x = kv + q / 2; y = pv + q / 2; r = norm(kv - pv);
                    dot(x, c(r) * (kv - pv) + c(norm(y)) * y) / dot(x, x))
    Cq(kv, pv, q) = ρ * B(norm(kv - pv)) * A(norm(kv - q / 2)) * A(norm(kv + q / 2)) *
                    U(kv, pv, q) * U(kv, pv, -q)

    # --- T1: closed form of C⁽²⁾ = (1/2d)∇²_q C_q against the Hessian ----------------
    rng = MersenneTwister(1)
    for d in 2:5, _ in 1:5
        kv, pv = 1.5 .* randn(rng, d), 1.5 .* randn(rng, d)
        k, p, r = norm(kv), norm(pv), norm(kv - pv)
        U0, V = MB.vertex_terms(k, p, r, c(r), c(p), D(c, p), D2(c, p), d)
        Λ  = (A(k) * D2(A, k) + (d - 1) * A(k) * D(A, k) / k - D(A, k)^2) / (4d)
        C2 = tr(ForwardDiff.hessian(q -> Cq(kv, pv, q), zero(kv))) / (2d)
        @test ρ * B(r) * A(k)^2 * U0^2 ≈ Cq(kv, pv, zero(kv)) rtol = 1e-12
        @test ρ * B(r) * (Λ * U0^2 + A(k)^2 / d * V) ≈ C2 rtol = 1e-10
    end

    # --- T2: U₀ = ṽ_k(p, k-p), and 2H is the corrected tr W of the SI ------------------
    for d in 2:5, _ in 1:3
        kv, pv = 1.5 .* randn(rng, d), 1.5 .* randn(rng, d)
        k, p, r = norm(kv), norm(pv), norm(kv - pv)
        μ = dot(kv, pv)
        U0, V = MB.vertex_terms(k, p, r, c(r), c(p), D(c, p), D2(c, p), d)
        @test U0 ≈ dot(kv, c(p) * pv + c(r) * (kv - pv)) / k^2
        G = ForwardDiff.gradient(q -> U(kv, pv, q), zero(kv))
        H = tr(ForwardDiff.hessian(q -> U(kv, pv, q), zero(kv)))
        @test V ≈ U0 * H - dot(G, G) rtol = 1e-10
        trW = ((d - 2) * (c(p) - U0) + D(c, p) * (p + (d + 1) * μ / (2p) - 2μ^2 / (p * k^2)) +
               μ / 2 * D2(c, p)) / k^2
        @test 2H ≈ trW rtol = 1e-10
        if d == 3   # Szamel's eq. (99), divided by k
            @test 2H ≈ (-U0 + c(p) + (p^2 + 2μ - 2(μ / k)^2) * D(c, p) / p + μ / 2 * D2(c, p)) / k^2 rtol = 1e-10
        end
    end

    # --- T3: k²⟨U₀H - |G|²⟩ → -2(d-1)/(d(d+2)) (p c′(p))² as k → 0 -----------------------
    for d in 2:4
        k, p = 1e-4, 1.3
        c1p, c2p = D(c, p), D2(c, p)
        θ = π .* ((1:2000) .- 0.5) ./ 2000                        # s = k̂·p̂ = cos θ
        w = sin.(θ) .^ (d - 2)                                    # (1-s²)^{(d-3)/2} ds
        r = @. sqrt(k^2 + p^2 - 2k * p * cos(θ))
        V = [MB.vertex_terms(k, p, r[n], c(r[n]), c(p), c1p, c2p, d)[2] for n in eachindex(θ)]
        @test k^2 * sum(w .* V) / sum(w) ≈ -2(d - 1) / (d * (d + 2)) * (p * c1p)^2 rtol = 1e-3
    end

    # --- T4: the :midpoint kernel is MCT.jl's discretisation ----------------------------
    for d in 2:4
        Nk = 40; Δk = 0.5; k = collect(Δk .* ((1:Nk) .- 0.5))
        S = @. 1 + 0.8sin(k) / k + 0.1exp(-k)
        F = rand(rng, Nk) .* S
        ref = ModeCouplingTheory.dDimModeCouplingKernel(0.8, 1.3, 0.7, k, S, d)
        for store in (true, false)
            bip = BipolarMCTKernel(0.8, 1.3, 0.7, k, S, d; store)
            @test evaluate_kernel(bip, F, 0.0).diag ≈ evaluate_kernel(ref, F, 0.0).diag rtol = 1e-12
        end
        if d == 3
            @test evaluate_kernel(BipolarMCTKernel(ref), F, 0.0).diag ≈
                  evaluate_kernel(ModeCouplingKernel(0.8, 1.3, 0.7, k, S), F, 0.0).diag rtol = 1e-12
        end
    end

    # --- T5: :anglecell integrates the angular measure exactly ---------------------------
    for d in 2:5
        Nk = 30; Δk = 0.25; k = collect(Δk .* ((1:Nk) .- 0.5))
        Ω = MB.surface_d_dim_unit_sphere
        err = 0.0
        for i in 1:Nk, j in 1:Nk-i+1                         # untruncated triangles
            lo, hi = MB.triangle_bounds(i, j, Nk)
            w̃ = sum(MB.angular_cdf(MB._cos_edge(k[i], k[j], l - 1, Δk), d) -
                    MB.angular_cdf(MB._cos_edge(k[i], k[j], l, Δk), d) for l in lo:hi)
            err = max(err, abs(w̃ - Ω(d) / Ω(d - 1)) / (Ω(d) / Ω(d - 1)))
        end
        @test err < 1e-14
        # the symmetrised weights of the kernel, stored or on the fly, are the same
        K1 = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), d; weights=:anglecell)
        K2 = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), d; weights=:anglecell, store=false)
        buf = zeros(Nk)
        @test all(MB.triangle_weights(K1, buf, i, j, MB.triangle_bounds(i, j, Nk)...) ≈
                  MB.triangle_weights(K2, buf, i, j, MB.triangle_bounds(i, j, Nk)...) for i in 1:Nk, j in 1:Nk)
    end
    # In d = 2, :midpoint samples the J^{-1/2} edge singularity and only approaches the
    # exact shell sum Σ_l w = Ω₂ p Δk/(2π)² as Δk → 0 (in d = 3 it is exact).
    function shell(Nk, d)
        Δk = 4 / Nk; k = collect(Δk .* ((1:Nk) .- 0.5)); i = j = Nk ÷ 4
        K = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), d; store=false)
        w = MB.triangle_weights(K, zeros(Nk), i, j, MB.triangle_bounds(i, j, Nk)...)
        return sum(w) / (MB.surface_d_dim_unit_sphere(d) * k[j]^(d - 1) * Δk / (2π)^d) - 1
    end
    @test abs(shell(80, 2)) < abs(shell(40, 2)) < abs(shell(20, 2))
    @test abs(shell(20, 3)) < 1e-14

    # --- Percus–Yevick hard spheres: Nk = 100, kmax = 40 -------------------------------
    Nk = 100; kmax = 40.0; Δk = kmax / Nk
    k = collect(Δk .* ((1:Nk) .- 0.5))
    template = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), 3)
    function system(φ)
        cφ, c1, c2 = dcf_PY(k, φ)
        return BipolarMCTKernel(template, 6φ / π, @. 1 / (1 - 6φ / π * cφ)), (c1, c2)
    end

    # c′, c″: the default quintic spline through the grid values vs the exact ones
    cη, c1η, c2η = dcf_PY(k, 0.515912)
    c1s, c2s = dcf_derivatives(k, cη)
    @test maximum(abs, c1s .- c1η) < 1e-4 * maximum(abs, c1η)
    @test maximum(abs, c2s .- c2η) < 1e-3 * maximum(abs, c2η)
    @test cη ≈ c_PY.(k, 0.515912) rtol = 1e-8               # BigFloat only fixes rounding

    # the critical point of the discrete plateau equation
    crit = find_critical_packing(system, 0.52)
    φc = crit.φc
    @test φc ≈ 0.51591 atol = 2e-5
    @test φc ≈ 0.5159121341 rtol = 1e-8                     # regression pin

    # a glass state close to it
    ε = 1e-5
    K, dc = system(φc * (1 + ε))
    F = steady_state(K)
    f = F ./ K.Sk
    @test maximum(abs, F .- steady_state_map(K, F)) < 1e-12

    # T6: C0 is the Jacobian of the plateau map; E0 → 1 like 2g(1-λ)√ε
    C0 = stability_matrix(K, F)
    @test C0 ≈ ForwardDiff.jacobian(x -> steady_state_map(K, x), F) rtol = 1e-10
    E0, ρR, ℓ = perron_pair(C0)
    @test (1 - E0) / sqrt(ε) ≈ crit.two_g_one_minus_λ rtol = 3e-3
    # MCT.jl's solver with the same kernel finds the same plateau
    sol = solve_steady_state(K.k_array .^ 2 ./ K.Sk, K.Sk, K; tolerance=1e-13, max_iterations=10^7)
    @test get_F(sol, 1, :) ≈ F rtol = 1e-8

    # T7: λ, normalisation-free, equals compute_lambda on the MCT.jl kernel
    λ, hR, hL, C1 = compute_lambda(K, f)
    kd = ModeCouplingTheory.dDimModeCouplingKernel(K.ρ, 1.0, 1.0, k, K.Sk, 3)
    λd, hRd, hLd, C1d = compute_lambda(kd, f)
    @test λ ≈ λd rtol = 1e-12
    @test hR ≈ hRd rtol = 1e-10
    @test hL ≈ hLd rtol = 1e-10
    @test λ ≈ 0.735 atol = 0.01

    # Γ
    G = compute_gamma(K, f; dc)
    @test G.E0 ≈ E0
    @test G.Γ ≈ G.Γ_A + G.Γ_V
    @test sum(G.profile) ≈ G.Γ
    @test G.Γ ≈ 0.0962333 rtol = 1e-5                        # regression pin, Nk=100, kmax=40
    @test compute_gamma(kd, f; dc).Γ ≈ G.Γ rtol = 1e-12      # via the MCT.jl kernel
    @test compute_gamma(K, f).Γ ≈ G.Γ rtol = 1e-4            # spline c′, c″

    # T8: the structure-factor part is diagonal, Γ_A = -E0 ⟨ℓ ρR Δ_d ln A⟩/(4d ⟨ℓ ρR⟩)
    A_ = K.Sk .- F
    A1, A2 = MB.even_derivatives(k, A_)
    ΔlnA = @. A2 / A_ - (A1 / A_)^2 + 2A1 / (k * A_)
    @test G.Γ_A ≈ -G.E0 * sum(G.ℓ .* G.ρR .* ΔlnA) / (12 * sum(G.ℓ .* G.ρR)) rtol = 1e-10

    # --- the ε → 0 limit and the literature -------------------------------------------
    rows = glass_sweep(system, φc, [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    sm = sweep_summary(rows)
    @test sm.two_g_one_minus_λ ≈ crit.two_g_one_minus_λ rtol = 2e-3
    @test sm.λc ≈ 0.735 atol = 0.005
    @test sm.Γc ≈ 0.09725 rtol = 1e-3                        # regression pin, Nk=100, kmax=40
    # Kuni's fit λ_q/λ_0 - 1 = 0.072 q² ε^{-1/2} is ξ₀² = Γ/(2g(1-λ)), not Γ; Γ itself is
    # ~35% larger than the 0.0708/0.072 quoted as "Γ" in the literature.
    @test sm.ξ0sq ≈ 0.072 rtol = 0.1
    @test !isapprox(sm.Γc, 0.072, rtol=0.2)

    # --- numerical PY for hard hyperspheres --------------------------------------------
    cn, c1n, c2n = dcf_PY_numeric(3, 0.5, k; M=200)
    ce, c1e, c2e = dcf_PY(k, 0.5)
    @test maximum(abs, cn .- ce) < 1e-6 * maximum(abs, ce)
    @test maximum(abs, c1n .- c1e) < 1e-6 * maximum(abs, c1e)
    @test maximum(abs, c2n .- c2e) < 1e-6 * maximum(abs, c2e)
    sol3 = solve_PY(3, 0.5; M=200)
    @test 1 / (1 - sol3.ρ * MB.dcf(sol3, 1e-6)[1]) ≈ S0_PY(0.5) rtol = 1e-3
    # d = 2: Richardson at M = 200 and M = 400 agree
    c2a = dcf_PY_numeric(2, 0.6, k; M=200)[1]
    c2b = dcf_PY_numeric(2, 0.6, k; M=400)[1]
    @test maximum(abs, c2a .- c2b) < 1e-5 * maximum(abs, c2b)
end

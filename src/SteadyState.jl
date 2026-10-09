using LinearAlgebra

"""
    steady_state(kernel::BipolarMCTKernel; F0=kernel.Sk, tol=1e-13, newton_switch=1e-6,
                 maxiter=10^6, liquid_tol=1e-10, verbose=false) -> F∞

Non-ergodicity parameter `F∞` (unnormalised, `F∞ = S f`) of the discrete plateau map `Φ(F) = S m(F)/(1 + m(F))` ([`steady_state_map`](@ref)); returns zeros in the liquid.

Picard iteration from `F0` (the map is monotone, so starting above — at `S`, or at the plateau of a denser state — converges from above) until successive iterates differ by less than `newton_switch`, then Newton, `F ← F - (I - C0)⁻¹(F - Φ(F))` with `C0` the [`stability_matrix`](@ref), until `‖F - Φ(F)‖∞ < tol`. Near the transition the Picard contraction rate is `E0 → 1`, which Newton does not feel; from above the fold Newton converges monotonically. A Newton step that does not reduce the residual is halved, and if that fails the solver falls back to Picard with a tighter switch.
"""
function steady_state(K::BipolarMCTKernel{T}; F0::AbstractVector=K.Sk, tol::Real=1e-13,
                      newton_switch::Real=1e-6, maxiter::Integer=10^6, liquid_tol::Real=1e-10,
                      verbose::Bool=false) where {T}
    S = K.Sk
    F = collect(T, F0)
    m = similar(F)
    it = 0
    switch = newton_switch
    while true
        # --- Picard ---------------------------------------------------------
        while true
            it += 1
            it > maxiter && error("steady_state: no convergence after $maxiter Picard iterations.")
            memory_functional!(m, K, F)
            Fn   = @. S * m / (1 + m)
            step = maximum(abs, Fn .- F)
            F    = Fn
            maximum(F ./ S) < liquid_tol && return zero(F)          # ergodic
            step < tol && return F
            step < switch && break
        end
        verbose && println("steady_state: Newton after $it Picard iterations.")
        # --- Newton ---------------------------------------------------------
        res = F .- steady_state_map(K, F)
        r   = norm(res, Inf)
        stalled = false
        for n in 1:50
            r < tol && return F
            δ = (I - stability_matrix(K, F)) \ res
            α = one(T)
            while true
                Ft   = F .- α .* δ
                rest = Ft .- steady_state_map(K, Ft)
                rt   = norm(rest, Inf)
                if rt < r
                    F, res, r = Ft, rest, rt
                    break
                end
                α /= 2
                if α < 1 / 64
                    stalled = true
                    break
                end
            end
            stalled && break
            verbose && println("steady_state: Newton $n, residual $r")
        end
        # Residual stuck at rounding level: accept.
        r < 100 * tol && return F
        switch /= 100
        verbose && println("steady_state: Newton stalled at residual $r, back to Picard.")
    end
end

_kernel_of(x) = x isa MemoryKernel ? x : first(x)

# 1 - E0 = α√x + βx + γx^{3/2}, x = φ - φc, is linear in (α, β, γ) at fixed φc; φc minimises the least-squares residual (variable projection): a scan of the bracket below the smallest φ, then golden section.
function _fold_fit(φs::AbstractVector, ys::AbstractVector)
    φmin, φmax = extrema(φs)
    function resid(φc)
        x = φs .- φc
        X = [sqrt.(x) x x .^ 1.5]
        return sum(abs2, X * (X \ ys) .- ys)
    end
    lo, hi = φmin - 3 * (φmax - φmin), φmin
    grid = range(lo, hi; length=402)[2:end-1]
    ib = argmin(resid.(grid))
    a, b = grid[max(ib - 1, 1)], grid[min(ib + 1, length(grid))]
    r = (sqrt(5) - 1) / 2
    for _ in 1:200
        b - a < 1e-15 * abs(b) && break
        x1, x2 = b - r * (b - a), a + r * (b - a)
        resid(x1) < resid(x2) ? (b = x2) : (a = x1)
    end
    φc = (a + b) / 2
    x  = φs .- φc
    α  = ([sqrt.(x) x x .^ 1.5] \ ys)[1]
    return φc, α
end

"""
    find_critical_packing(system, φ_glass; εs=(1e-4, 5e-5, 2.5e-5, 1.25e-5),
                          εs_fine=(1e-4, ..., 1.5625e-6), rtol=1e-9, maxrounds=8,
                          verbose=false) -> (; φc, two_g_one_minus_λ)

Critical packing fraction of the discrete plateau equation. `system(φ)` must return the [`BipolarMCTKernel`](@ref) at packing fraction `φ` (or a tuple whose first element is it), always on the same grid; `φ_glass` must lie in the glass.

On the glass side `1 - E0 = 2g(1-λ)√ε + O(ε)`, `ε = (φ - φc)/φc`. Each round solves the plateau at `φ = φc_estimate·(1 + ε)` for the given `εs` (by continuation from the nearest denser solution) and extrapolates the straight-line fit of `(1 - E0)²` to zero, until the estimate moves by less than `rtol`. That fit is biased by the `O(ε^{3/2})` term, so a last round over `εs_fine` fits `1 - E0 = α√(φ-φc) + β(φ-φc) + γ(φ-φc)^{3/2}`, nonlinear in `φc` only. Returns `φc` and `2g(1-λ) = α√φc`.
"""
function find_critical_packing(system, φ_glass::Real; εs=(1e-4, 5e-5, 2.5e-5, 1.25e-5),
                               εs_fine=1e-4 ./ 2 .^ (0:6), rtol::Real=1e-9,
                               maxrounds::Integer=8, verbose::Bool=false)
    sols = Dict{Float64,Vector{Float64}}()             # φ => F∞, glass states only
    function E0_at(φ)
        K = _kernel_of(system(φ))
        denser = filter(>(φ), collect(keys(sols)))
        F0 = isempty(denser) ? K.Sk : sols[minimum(denser)]
        F = steady_state(K; F0)
        all(iszero, F) && return nothing
        sols[φ] = F
        E0, _, _ = perron_pair(stability_matrix(K, F))
        verbose && println("find_critical_packing: φ = $φ, 1 - E0 = $(1 - E0)")
        return E0
    end

    Eg = E0_at(φ_glass)
    Eg === nothing && error("find_critical_packing: φ_glass = $φ_glass is not in the glass.")
    # Second point for a first secant estimate of the zero of (1 - E0)².
    δ, E2 = 0.25 * (1 - Eg)^2, nothing
    while E2 === nothing
        E2 = E0_at(φ_glass * (1 - δ))
        E2 === nothing && (δ /= 4)
    end
    φ2 = φ_glass * (1 - δ)
    slope = ((1 - Eg)^2 - (1 - E2)^2) / (φ_glass - φ2)
    φc = φ2 - (1 - E2)^2 / slope

    for round in 1:maxrounds
        pts = Tuple{Float64,Float64}[]
        for ε in sort(collect(float.(εs)); rev=true)
            E = E0_at(φc * (1 + ε))
            E === nothing || push!(pts, (φc * (1 + ε), (1 - E)^2))
        end
        if length(pts) < 2               # estimate fell into the liquid: move up
            φc *= 1 + 4 * maximum(εs)
            continue
        end
        X = [ones(length(pts)) first.(pts)]
        a, b = X \ last.(pts)
        φnew = -a / b
        verbose && println("find_critical_packing: round $round, φc = $φnew")
        converged = abs(φnew - φc) < rtol * φc
        φc = φnew
        converged && break
    end

    φs, ys = Float64[], Float64[]
    for ε in sort(collect(float.(εs_fine)); rev=true)
        φ = φc * (1 + ε)
        E = E0_at(φ)
        E === nothing || (push!(φs, φ); push!(ys, 1 - E))
    end
    @assert length(φs) ≥ 5 "find_critical_packing: too few glass states for the final fit."
    φc, α = _fold_fit(φs, ys)
    verbose && println("find_critical_packing: final fit, φc = $φc")
    return (; φc, two_g_one_minus_λ=α * sqrt(φc))
end

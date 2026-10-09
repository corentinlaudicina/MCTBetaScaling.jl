using LinearAlgebra
using ModeCouplingTheory
import ModeCouplingTheory: dDimModeCouplingKernel

# ---------------------------------------------------------------------------
# Cubic-spline derivatives on an equidistant grid.
#
# "Not-a-knot" end conditions rather than natural ones, which would force a
# vanishing second derivative at the first grid point.
# ---------------------------------------------------------------------------
"""
    spline_derivatives(x, y) -> (y′, y″)

First and second derivatives of the not-a-knot interpolating cubic spline
through `(x, y)`, evaluated at the nodes `x`. `x` must be equidistant.
"""
function spline_derivatives(x::AbstractVector, y::AbstractVector)
    M = _spline_moments(x, y)
    N = length(x)
    h = x[2] - x[1]
    # S'(x_i) from the cubic on [x_i, x_{i+1}] (last node from the preceding cell).
    yp = zeros(Float64, N)
    @inbounds for i in 1:N-1
        yp[i] = (y[i+1] - y[i]) / h - h * (2M[i] + M[i+1]) / 6
    end
    yp[N] = (y[N] - y[N-1]) / h + h * (M[N-1] + 2M[N]) / 6
    return yp, M
end

# Moments M_i = S''(x_i): tridiagonal system from C² continuity of the spline,
# closed by the not-a-knot conditions M₁ - 2M₂ + M₃ = 0 (and its mirror image),
# which make the third derivative continuous across the second/second-to-last knot.
function _spline_moments(x::AbstractVector, y::AbstractVector)
    N = length(x)
    @assert N == length(y) "x and y must have the same length."
    @assert N ≥ 4 "The not-a-knot spline needs at least 4 grid points."
    h = x[2] - x[1]
    @assert all(diff(x) .≈ h) "The grid must be equidistant."
    A = zeros(Float64, N, N)
    r = zeros(Float64, N)
    @inbounds for i in 2:N-1
        A[i, i-1] = 1.0
        A[i, i]   = 4.0
        A[i, i+1] = 1.0
        r[i] = 6 * (y[i-1] - 2y[i] + y[i+1]) / h^2
    end
    A[1, 1], A[1, 2], A[1, 3] = 1.0, -2.0, 1.0
    A[N, N-2], A[N, N-1], A[N, N] = 1.0, -2.0, 1.0
    return A \ r
end

# Finite-difference weights w with Σ_s w_s f(x + o_s h) ≈ hᵐ f⁽ᵐ⁾(x).
function _fd_weights(offsets, m::Integer)
    n = length(offsets)
    V = [float(o)^p for p in 0:n-1, o in offsets]
    e = zeros(n)
    e[m+1] = factorial(m)
    return V \ e
end

"""
    even_derivatives(k_array, y; scheme=:fd4) -> (y′, y″)

First and second derivatives on the midpoint grid `k_i = (i - ½)Δk` of a function that
is even in `k`, through the even extension `y(-k₁) = y(k₁)`, `y(-k₂) = y(k₂)`.

* `:fd4`: fourth-order central differences, and six-point one-sided stencils at the two
  largest `k`;
* `:spline`: the not-a-knot cubic spline through the even extension
  ([`spline_derivatives`](@ref)).
"""
function even_derivatives(k::AbstractVector, y::AbstractVector; scheme::Symbol=:fd4)
    N = length(y)
    h = k[2] - k[1]
    if scheme === :spline
        x, ye = _even_extension(collect(float.(k)), collect(float.(y)), 3)
        yp, ypp = spline_derivatives(x, ye)
        return yp[4:end], ypp[4:end]
    end
    @assert scheme === :fd4 "scheme must be :fd4 or :spline."
    @assert N ≥ 6 "even_derivatives needs at least 6 grid points."
    ye(i) = y[i ≥ 1 ? i : 1 - i]                      # k_{1-i} = -k_i
    d1 = (1, -8, 0, 8, -1) ./ 12
    d2 = (-1, 16, -30, 16, -1) ./ 12
    y1 = zeros(float(eltype(y)), N)
    y2 = zeros(float(eltype(y)), N)
    for i in 1:N-2
        y1[i] = sum(d1[s+3] * ye(i + s) for s in -2:2) / h
        y2[i] = sum(d2[s+3] * ye(i + s) for s in -2:2) / h^2
    end
    for (i, offs) in ((N - 1, -4:1), (N, -5:0))
        w1, w2 = _fd_weights(offs, 1), _fd_weights(offs, 2)
        y1[i] = sum(w1[n] * y[i+o] for (n, o) in enumerate(offs)) / h
        y2[i] = sum(w2[n] * y[i+o] for (n, o) in enumerate(offs)) / h^2
    end
    return y1, y2
end

# ---------------------------------------------------------------------------
# The q-dependent stability operator.
#
#   (C_q χ)(k) = ρ ∫dᵈp/(2π)ᵈ A(|k-q/2|) A(|k+q/2|) U(q) U(-q) B(|k-p|) χ(p),
#   U(q) = ṽ_{k+q/2}(k-p, p+q/2) = (k+q/2)·[c(|k-p|)(k-p) + c(|p+q/2|)(p+q/2)]/|k+q/2|²,
#
# with A = S - F∞ = S(1-f) and B = F∞ = S f. C_q = C_{-q}, so C_q = C₀ + q² C⁽²⁾ + O(q⁴)
# on isotropic functions, with C⁽²⁾ = (1/2d) ∇²_q C_q at q = 0. Both A(|k∓q/2|)A(|k±q/2|)
# and U(q)U(-q) are even in q, so their gradients vanish at q = 0 and
#
#   C⁽²⁾(k,p,r) = ρ B(r) [Λ(k) U₀² + (A(k)²/d)(U₀ H - |G|²)],
#   Λ = [A A″ + (d-1) A A′/k - A′²]/4d = (A²/4d) Δ_d ln A ,
#
# where U₀ = U(0), G = ∇_q U and H = ∇²_q U at q = 0 (the paper's w = 2G, tr W = 2H).
# ---------------------------------------------------------------------------
"""
    vertex_terms(k, p, r, cr, cp, dcp, ddcp, d) -> (U₀, U₀H - |G|²)

Vertex pieces of the `q²` coefficient on the triangle `(k, p, r = |k-p|)`, with
`cr = c(r)`, `cp = c(p)`, `dcp = c′(p)`, `ddcp = c″(p)`; `p` is the leg that carries
the perturbation and the derivatives of `c` are taken there. `C₀ ∝ U₀²` and
`C⁽²⁾_V ∝ (A²/d)(U₀H - |G|²)`.
"""
@inline function vertex_terms(k, p, r, cr, cp, dcp, ddcp, d)
    μ  = (k^2 + p^2 - r^2) / 2
    N0 = cr * (k^2 - μ) + cp * μ
    U0 = N0 / k^2
    gk = (cr + cp) / (2k^2) - N0 / k^4
    gp = (cp - cr + dcp * μ / p) / (2k^2)
    G2 = gk^2 * k^2 + 2gk * gp * μ + gp^2 * p^2
    H  = (d / 2 - 1) * (cp / k^2 - N0 / k^4) +
         dcp * (p / (2k^2) + (d + 1) * μ / (4p * k^2) - μ^2 / (p * k^4)) +
         ddcp * μ / (4k^2)
    return U0, U0 * H - G2
end

"""
    gamma_matrices(kernel::BipolarMCTKernel, F, dc; Aderiv=:fd4) -> (; C0, C2A, C2V, Λ)

Stability matrix `C0` (see [`stability_matrix`](@ref)) and the two parts of its `q²`
coefficient on the kernel's weights,

    C2V[i,j] = ρ A_i²/d Σ_l w_{ijl} B_l (U₀H - |G|²)(k_i, p_j, r_l)
    C2A[i,j] = (Λ_i/A_i²) C0[i,j] ,

at the plateau `F` (unnormalised). `dc = (c′, c″)` on the grid; `Λ` uses `A′, A″` from
[`even_derivatives`](@ref) with `scheme = Aderiv`.
"""
function gamma_matrices(K::BipolarMCTKernel{T}, F::AbstractVector, dc; Aderiv::Symbol=:fd4) where {T}
    Nk, d, k, C = K.Nk, K.d, K.k_array, K.C
    c1, c2 = dc
    @assert length(F) == Nk && length(c1) == Nk && length(c2) == Nk "F, c′ and c″ must live on the kernel grid."
    A = K.Sk .- F
    A1, A2 = even_derivatives(k, A; scheme=Aderiv)
    Λ = @. (A * A2 + (d - 1) * A * A1 / k - A1^2) / (4d)

    C0  = zeros(T, Nk, Nk)
    C2V = zeros(T, Nk, Nk)
    Threads.@threads for i in 1:Nk
        buf = Vector{T}(undef, Nk)
        ki  = k[i]
        for j in 1:Nk
            lo, hi = triangle_bounds(i, j, Nk)
            w  = triangle_weights(K, buf, i, j, lo, hi)
            pj, cp, dcp, ddcp = k[j], C[j], c1[j], c2[j]
            s0 = sv = zero(T)
            @inbounds for l in lo:hi
                U0, V = vertex_terms(ki, pj, k[l], C[l], cp, dcp, ddcp, d)
                wb  = w[l-lo+1] * F[l]
                s0 += wb * U0^2
                sv += wb * V
            end
            C0[i, j]  = K.ρ * A[i]^2 * s0
            C2V[i, j] = K.ρ * A[i]^2 / d * sv
        end
    end
    C2A = Diagonal(Λ ./ A .^ 2) * C0
    return (; C0, C2A, C2V, Λ)
end

"""
    compute_gamma(kernel, f; dc=dcf_derivatives(kernel.k_array, kernel.C), Aderiv=:fd4)
        -> (; Γ, Γ_A, Γ_V, E0, profile, ρR, ℓ)

Coefficient `Γ` of the Laplacian term of the (stochastic) β-scaling equation

    σ(x) - δ t + λ g(x,t)² + Γ ∇²g(x,t) = ∂ₜ∫ g(x,t-τ) g(x,τ) dτ ,

i.e. the `q²` coefficient of the top eigenvalue of the MCT stability operator at finite
wavevector, `E0(q) = E0 - Γ q² + O(q⁴)`. `Γ` carries units of σ² in the grid units.
`kernel` is a [`BipolarMCTKernel`](@ref) or a `dDimModeCouplingKernel` (which is
handled as the equivalent `:midpoint` kernel).

`f` is the *normalized* non-ergodicity parameter `f(k) = F∞(k)/S(k)`, as in
[`compute_lambda`](@ref); it must be the plateau of the same discrete kernel, otherwise
`E0 ≠ 1` at the discrete critical point. `dc = (c′, c″)` are the derivatives of the
direct correlation function on the grid: by default from a quintic spline through
`kernel.C` ([`dcf_derivatives`](@ref)); for PY hard spheres pass the exact ones,
`dcf_PY(k, η)[2:3]`.

# Method

The operator is the symmetric parametrisation of the IMCT stability ("mass") operator
(see the comment block in src/Gamma.jl); `Γ` follows by first-order perturbation theory
on the grid,

    Γ = -ℓᵀ C⁽²⁾ ρR / ℓᵀ ρR ,   C⁽²⁾ = C2A + C2V ([`gamma_matrices`](@ref)),

with `ρR`, `ℓ` the right and left Perron vectors of `C0`. The ratio fixes the first
Götze normalisation and does not depend on the second one, so no normalisation enters.
`C_q = C_{-q}` leaves no `O(q)` term, and the `l = 2` mixing enters only at `O(q⁴)`.

At the plateau `1 - E0 = 2g(1-λ)√ε`, so the correlation length of the plateau
fluctuations is `ξ² = Γ/(1 - E0)`, i.e. `ξ = ξ₀ ε^{-1/4}` with `ξ₀ = √(Γ/(2g(1-λ)))`;
see [`glass_sweep`](@ref).

# Returns

`Γ` and its structure-factor (`Γ_A`, from `Λ`) and vertex (`Γ_V`) parts, the top
eigenvalue `E0` of `C0`, the per-`k` contributions `profile` (`sum(profile) = Γ`) and
the Perron vectors `ρR`, `ℓ` (unit sum). In `d = 2` the integrand of `Γ` behaves like
`1/k` as `k → 0`, so there `Γ` depends logarithmically on the grid cutoff `k₁ = Δk/2`.
"""
function compute_gamma(K::BipolarMCTKernel, f::AbstractVector;
                       dc=dcf_derivatives(K.k_array, K.C), Aderiv::Symbol=:fd4)
    @assert length(f) == K.Nk "length(f) must match the kernel grid (kernel.Nk = $(K.Nk))."
    @assert all(x -> x < 1, f) "f(k) must be < 1 on the whole grid."
    M = gamma_matrices(K, K.Sk .* f, dc; Aderiv)
    E0, ρR, ℓ = perron_pair(M.C0)
    den = dot(ℓ, ρR)
    vA  = M.C2A * ρR
    vV  = M.C2V * ρR
    Γ_A = -dot(ℓ, vA) / den
    Γ_V = -dot(ℓ, vV) / den
    profile = @. -ℓ * (vA + vV) / den
    return (; Γ=Γ_A + Γ_V, Γ_A, Γ_V, E0, profile, ρR, ℓ)
end

compute_gamma(kernel::dDimModeCouplingKernel, f::AbstractVector; kwargs...) =
    compute_gamma(BipolarMCTKernel(kernel), f; kwargs...)

# ---------------------------------------------------------------------------
# Glass-side sweeps.
# ---------------------------------------------------------------------------
"""
    glass_sweep(system, φc, εs; Aderiv=:fd4) -> Vector{NamedTuple}

Plateau, `λ` and `Γ` along `φ = φc (1 + ε)` for `ε ∈ εs`, solved from the largest `ε`
down by continuation. `system(φ)` returns `(kernel, (c′, c″))` on a fixed grid. Each row
holds `ε`, `φ`, `E0`, `one_minus_E0`, `ratio = (1 - E0)/√ε` (→ `2g(1-λ)`), `λ`,
`K = Γ`, `K_A`, `K_V` and `xi0sq = ξ²√ε = Γ√ε/(1 - E0)` (→ `ξ₀²`).
"""
function glass_sweep(system, φc::Real, εs; Aderiv::Symbol=:fd4, verbose::Bool=false)
    rows = []
    F = nothing
    for ε in sort(collect(float.(εs)); rev=true)
        φ = φc * (1 + ε)
        K, dc = system(φ)
        F = steady_state(K; F0=F === nothing ? K.Sk : F)
        all(iszero, F) && error("glass_sweep: φ = $φ (ε = $ε) is not in the glass.")
        f = F ./ K.Sk
        G = compute_gamma(K, f; dc, Aderiv)
        λ = first(compute_lambda(K, f))
        ome = 1 - G.E0
        row = (ε=ε, φ=φ, E0=G.E0, one_minus_E0=ome, ratio=ome / sqrt(ε), λ=λ,
               K=G.Γ, K_A=G.Γ_A, K_V=G.Γ_V, xi0sq=G.Γ * sqrt(ε) / ome)
        verbose && println(row)
        push!(rows, row)
    end
    return [r for r in rows]
end

_lsq_powers(x, y, powers) = [xi^p for xi in x, p in powers] \ y

"""
    sweep_summary(rows; εmin=0, εmax=1e-3) -> (; Γc, two_g_one_minus_λ, λc, g, ξ0, ξ0sq)

Critical values from a [`glass_sweep`](@ref), using the rows with `εmin ≤ ε ≤ εmax`, all
expanded in powers of `√ε`: `2g(1-λ)` from `1 - E0 = 2g(1-λ)√ε + bε + cε^{3/2}`, and
`Γc`, `λc` from `X = Xc + a√ε + bε + cε^{3/2}`. Then `ξ₀ = √(Γc/(2g(1-λ)))` and
`g = 2g(1-λ)/(2(1-λc))`.
"""
function sweep_summary(rows; εmin::Real=0, εmax::Real=1e-3)
    sel = [r for r in rows if εmin <= r.ε <= εmax]
    @assert length(sel) ≥ 5 "sweep_summary needs at least five rows in [εmin, εmax]."
    ε = [r.ε for r in sel]
    α  = _lsq_powers(ε, [r.one_minus_E0 for r in sel], (0.5, 1.0, 1.5))[1]
    Γc = _lsq_powers(ε, [r.K for r in sel], (0.0, 0.5, 1.0, 1.5))[1]
    λc = _lsq_powers(ε, [r.λ for r in sel], (0.0, 0.5, 1.0, 1.5))[1]
    return (; Γc, two_g_one_minus_λ=α, λc, g=α / (2 * (1 - λc)), ξ0=sqrt(Γc / α), ξ0sq=Γc / α)
end

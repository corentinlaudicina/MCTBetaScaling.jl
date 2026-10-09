using LinearAlgebra
using ModeCouplingTheory
import ModeCouplingTheory: MemoryKernel, evaluate_kernel!, evaluate_kernel, dDimModeCouplingKernel, surface_d_dim_unit_sphere
import SpecialFunctions: beta_inc, beta

# ---------------------------------------------------------------------------
# Bipolar quadrature on the MCT grid.
#
# For any g(p, |k-p|),
#
#   ∫dᵈp/(2π)ᵈ g = Ω_{d-1} 2^{3-d}/(2π)ᵈ k^{2-d} ∫dp ∫_{|k-p|}^{k+p} dr p r J^{(d-3)/2} g(p, r)
#                = Ω_{d-1}/(2π)ᵈ ∫dp p^{d-1} ∫_{-1}^{1} ds (1-s²)^{(d-3)/2} g ,
#
# J = 4k²p² - (k²+p²-r²)², s = (k²+p²-r²)/2kp, Ω_{d-1} = surface_d_dim_unit_sphere(d-1).
# On the grid k_i = (i - ½)Δk the legs p = k_j, r = k_l close a triangle with k_i iff
# |i-j|+1 ≤ l ≤ i+j-1, and the grid truncates l ≤ Nk. The cell edges of r are then
# exactly |k_i - p_j| and k_i + p_j.
# ---------------------------------------------------------------------------
@inline triangle_bounds(i, j, Nk) = (abs(i - j) + 1, min(i + j - 1, Nk))

# J^{(d-3)/2}, without a pow call in the common dimensions.
@inline function _Jpow(k, p, r, d)
    d == 3 && return one(k)
    J = 4k^2 * p^2 - (k^2 + p^2 - r^2)^2
    d == 2 && return 1 / sqrt(J)
    d == 4 && return sqrt(J)
    return J^((d - 3) / 2)
end

"""
    angular_cdf(s, d)

`G_d(s) = ∫_{-1}^{s} (1-t²)^{(d-3)/2} dt`, the cumulative angular measure between two wavevectors in `d` dimensions; `G_d(1) = Ω_d/Ω_{d-1}`.
"""
@inline function angular_cdf(s, d)
    d == 2 && return asin(s) + π / 2
    d == 3 && return 1 + s
    d == 4 && return (s * sqrt(1 - s^2) + asin(s)) / 2 + π / 4
    a = (d - 1) / 2
    return 2.0^(d - 2) * beta_inc(a, a, (1 + s) / 2)[1] * beta(a, a)
end

# cos∠(k, p) on the edge r = mΔk of the r-cells.
@inline _cos_edge(k, p, m, Δk) = clamp((k^2 + p^2 - (m * Δk)^2) / (2k * p), -one(k), one(k))

"""
    BipolarMCTKernel(ρ, kBT, m, k_array, Sₖ, d; weights=:midpoint, store=length(k_array) ≤ 250)
    BipolarMCTKernel(kernel::dDimModeCouplingKernel)

One-component mode-coupling memory kernel in `d` dimensions,

    K(k) = ρ kBT/(2m) ∫dᵈp/(2π)ᵈ V(k,p)² F(p) F(|k-p|),   V = k̂·[p c(p) + (k-p) c(|k-p|)] ,

discretised with the bipolar quadrature `∫dᵈp/(2π)ᵈ g → Σ_{j,l} w_{ijl} g(p_j, r_l)` on the midpoint grid `k_array` (`k₁ = Δk/2`), over the triangle `|i-j|+1 ≤ l ≤ min(i+j-1, Nk)`. It is a `ModeCouplingTheory.MemoryKernel`, so `solve_steady_state` works with it, and [`compute_gamma`](@ref), [`compute_lambda`](@ref) and [`steady_state`](@ref) use the same weights — the stability matrix is then the exact Jacobian of the discrete plateau map.

# Weights

* `:midpoint`: `w_{ijl} = Ω_{d-1} 2^{3-d}/(2π)ᵈ k_i^{2-d} p_j r_l J^{(d-3)/2} Δk²`, the discretisation of `dDimModeCouplingKernel` (the two kernels agree to rounding).
* `:anglecell`: the angular measure is integrated exactly over each `r`-cell at `p = p_j`, `w̃_{ijl} = Ω_{d-1}/(2π)ᵈ p_j^{d-1} Δk [G_d(s_{l-1}) - G_d(s_l)]` with `G_d` the [`angular_cdf`](@ref) and `s_m` the cosine on the cell edge `r = mΔk`, then symmetrised, `w = (w̃_{ijl} + w̃_{ilj})/2`. In `d = 2` this integrates the `J^{-1/2}` edge singularity that `:midpoint` samples.

`store = true` keeps the weights in a packed triangle (`≈ Nk³/2` numbers); otherwise they are recomputed on the fly.
"""
struct BipolarMCTKernel{T} <: MemoryKernel
    d::Int
    ρ::T
    kBT::T
    m::T
    Nk::Int
    k_array::Vector{T}
    Δk::T
    Sk::Vector{T}
    C::Vector{T}                 # c(k) = (1 - 1/S(k))/ρ
    weights::Symbol
    offsets::Matrix{Int}         # start of the (i,j) segment in W; 0×0 when not stored
    W::Vector{T}
end

function BipolarMCTKernel(ρ, kBT, m, k_array, Sₖ, d::Integer; weights::Symbol=:midpoint,
                          store::Bool=length(k_array) <= 250)
    T  = float(promote_type(typeof(ρ), typeof(kBT), typeof(m), eltype(k_array), eltype(Sₖ)))
    k  = collect(T, k_array)
    S  = collect(T, Sₖ)
    Nk = length(k)
    Δk = k[2] - k[1]
    @assert length(S) == Nk "k_array and Sₖ must have the same length."
    @assert k[1] ≈ Δk / 2 "The first grid point must be exactly half the grid spacing."
    @assert all(diff(k) .≈ Δk) "The grid must be equidistant."
    @assert weights in (:midpoint, :anglecell) "weights must be :midpoint or :anglecell."
    @assert d >= 2 "d must be at least 2."
    C = @. (1 - 1 / S) / T(ρ)
    K = BipolarMCTKernel{T}(d, T(ρ), T(kBT), T(m), Nk, k, Δk, S, C, weights, zeros(Int, 0, 0), T[])
    store || return K

    offsets = zeros(Int, Nk, Nk)
    n = 0
    for j in 1:Nk, i in 1:Nk
        lo, hi = triangle_bounds(i, j, Nk)
        offsets[i, j] = n
        n += hi - lo + 1
    end
    W = Vector{T}(undef, n)
    Threads.@threads for i in 1:Nk
        for j in 1:Nk
            lo, hi = triangle_bounds(i, j, Nk)
            _weights!(view(W, offsets[i, j] .+ (1:hi-lo+1)), K, i, j, lo, hi)
        end
    end
    return BipolarMCTKernel{T}(d, T(ρ), T(kBT), T(m), Nk, k, Δk, S, C, weights, offsets, W)
end

BipolarMCTKernel(kernel::dDimModeCouplingKernel; kwargs...) =
    BipolarMCTKernel(kernel.ρ, kernel.kBT, kernel.m, kernel.k_array, kernel.Sk, kernel.d;
                     weights=:midpoint, kwargs...)

"""
    BipolarMCTKernel(template::BipolarMCTKernel, ρ, Sₖ)

Kernel at a new density and structure factor on the grid of `template`, sharing (not
copying) its stored weights, which depend on the grid only.
"""
function BipolarMCTKernel(K::BipolarMCTKernel{T}, ρ, Sₖ) where {T}
    S = collect(T, Sₖ)
    @assert length(S) == K.Nk "Sₖ must live on the grid of the template kernel."
    C = @. (1 - 1 / S) / T(ρ)
    return BipolarMCTKernel{T}(K.d, T(ρ), K.kBT, K.m, K.Nk, K.k_array, K.Δk, S, C, K.weights,
                               K.offsets, K.W)
end

# Fill buf[1:hi-lo+1] with w_{ijl}, l = lo:hi.
function _weights!(buf, K::BipolarMCTKernel{T}, i, j, lo, hi) where {T}
    d, Δk, k = K.d, K.Δk, K.k_array
    ki, pj = k[i], k[j]
    Ω = T(surface_d_dim_unit_sphere(d - 1))
    if K.weights === :midpoint
        P = Ω * T(2)^(3 - d) / T(2π)^d * Δk^2 / ki^(d - 2)
        @inbounds for l in lo:hi
            rl = k[l]
            buf[l-lo+1] = P * pj * rl * _Jpow(ki, pj, rl, d)
        end
    else
        Q = Ω / T(2π)^d * Δk / 2                       # the ½ of the symmetrisation
        a = Q * pj^(d - 1)
        Gprev = angular_cdf(_cos_edge(ki, pj, lo - 1, Δk), d)
        @inbounds for l in lo:hi
            Gl = angular_cdf(_cos_edge(ki, pj, l, Δk), d)
            rl = k[l]                                   # w̃_{ilj}: p-leg r_l, r-cell j
            Gj = angular_cdf(_cos_edge(ki, rl, j - 1, Δk), d) - angular_cdf(_cos_edge(ki, rl, j, Δk), d)
            buf[l-lo+1] = a * (Gprev - Gl) + Q * rl^(d - 1) * Gj
            Gprev = Gl
        end
    end
    return buf
end

# w_{ijl}, l = lo:hi, as a view: into the packed store, or into `buf` after filling it.
@inline function triangle_weights(K::BipolarMCTKernel, buf, i, j, lo, hi)
    n = hi - lo + 1
    isempty(K.W) && return view(_weights!(buf, K, i, j, lo, hi), 1:n)
    return view(K.W, K.offsets[i, j]+1:K.offsets[i, j]+n)
end

# k² U₀ with U₀ = ṽ_k(p, k-p) = k·[c(r)(k-p) + c(p)p]/k², the bare MCT vertex.
@inline function _N0(k2, p2, r2, cr, cp)
    μ = (k2 + p2 - r2) / 2
    return cr * (k2 - μ) + cp * μ
end

"""
    memory_functional(kernel::BipolarMCTKernel, F) -> m

`m(k) = K(k)/γ(k) = (ρ S(k)/2) ∫dᵈp/(2π)ᵈ U₀² F(p) F(|k-p|)`, `γ = kBT k²/(m S)`, with
`U₀ = V/k` the vertex of [`BipolarMCTKernel`](@ref); `m = f/(1-f)` at the plateau.
Generic in the element type of `F` (ForwardDiff-compatible).
"""
memory_functional(K::BipolarMCTKernel{T}, F::AbstractVector) where {T} =
    memory_functional!(similar(F, promote_type(T, eltype(F))), K, F)

function memory_functional!(out::AbstractVector, K::BipolarMCTKernel{T}, F::AbstractVector) where {T}
    Nk, k, C = K.Nk, K.k_array, K.C
    @assert length(F) == Nk
    Threads.@threads for i in 1:Nk
        buf = Vector{T}(undef, Nk)
        k2  = k[i]^2
        acc = zero(eltype(out))
        for j in 1:Nk
            lo, hi = triangle_bounds(i, j, Nk)
            w  = triangle_weights(K, buf, i, j, lo, hi)
            p2 = k[j]^2
            cp = C[j]
            s  = zero(eltype(out))
            @inbounds @simd for l in lo:hi
                N0 = _N0(k2, p2, k[l]^2, C[l], cp)
                s += w[l-lo+1] * N0^2 * F[l]
            end
            acc += s * F[j]
        end
        out[i] = K.ρ * K.Sk[i] / 2 * acc / k2^2
    end
    return out
end

"""
    steady_state_map(kernel::BipolarMCTKernel, F) -> S m(F)/(1 + m(F))

The plateau map whose fixed point is the non-ergodicity parameter `F∞`, i.e. what
`solve_steady_state` iterates, `F = (K + γ)⁻¹ K S`.
"""
function steady_state_map(K::BipolarMCTKernel, F::AbstractVector)
    m = memory_functional(K, F)
    return @. K.Sk * m / (1 + m)
end

function evaluate_kernel!(out::Diagonal, K::BipolarMCTKernel, F::AbstractVector, t)
    memory_functional!(out.diag, K, F)
    @. out.diag *= K.kBT * K.k_array^2 / (K.m * K.Sk)        # K = γ m
    return out
end

function evaluate_kernel(K::BipolarMCTKernel, F::AbstractVector, t)
    out = Diagonal(similar(F, promote_type(eltype(K.Sk), eltype(F))))
    return evaluate_kernel!(out, K, F, t)
end

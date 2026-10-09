using LinearAlgebra
using ForwardDiff
import Dierckx
import SpecialFunctions: besselj, besselj0, besselj1, gamma
import ModeCouplingTheory: volume_d_dim_sphere

# ---------------------------------------------------------------------------
# Percus–Yevick hard spheres in d = 3 (Wertheim's closed form).
# ---------------------------------------------------------------------------
"""
    c_PY(k, η)

Percus–Yevick direct correlation function `c(k)` of hard spheres of unit diameter at packing fraction `η` (Wertheim's closed form), with `n c = 1 - 1/S`.

Generic in the number type, so that it can be differentiated with ForwardDiff and evaluated in `BigFloat`. The closed form cancels like `1/k⁶` as `k → 0`, so in Float64 it loses about `6 log₁₀(1/k)` digits there, and its derivatives lose more; on a grid use [`dcf_PY`](@ref), which switches to `BigFloat` at small `k`.
"""
function c_PY(k, η)
    A = -(1 - η)^-4 * (1 + 2η)^2
    B = (1 - η)^-4 * 6η * (1 + η / 2)^2
    D = -(1 - η)^-4 * η * (1 + 2η)^2 / 2
    # π/k⁶ first, so that π is promoted to the type of k (BigFloat, Dual, ...).
    return 4 * (π / k^6) * (24D - 2B * k^2 - (24D - 2(B + 6D) * k^2 + (A + B + D) * k^4) * cos(k) +
                            k * (-24D + (A + 2B + 4D) * k^2) * sin(k))
end

"""
    S_PY(k, η)

Percus–Yevick structure factor of hard spheres, `S = 1/(1 - n c_PY)`, `n = 6η/π`.
"""
S_PY(k, η) = 1 / (1 - (6η / π) * c_PY(k, η))

_taylor2(f, x) = (f(x), ForwardDiff.derivative(f, x),
                  ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x))

"""
    dcf_PY(k_array, η; bigfloat_below=2.0, precision=256) -> (c, c′, c″)

`c_PY` and its first two derivatives on `k_array`, by forward-mode differentiation of the closed form. Points with `k < bigfloat_below` are evaluated in `BigFloat` of the given `precision`, which removes the small-`k` cancellation of the closed form.
"""
function dcf_PY(k::AbstractVector, η::Real; bigfloat_below::Real=2.0, precision::Integer=256)
    vals = map(k) do x
        if x < bigfloat_below
            setprecision(BigFloat, precision) do
                ηb = BigFloat(η)
                Float64.(_taylor2(y -> c_PY(y, ηb), BigFloat(x)))
            end
        else
            _taylor2(y -> c_PY(y, η), float(x))
        end
    end
    return getindex.(vals, 1), getindex.(vals, 2), getindex.(vals, 3)
end

# ---------------------------------------------------------------------------
# Derivatives of a tabulated direct correlation function.
# ---------------------------------------------------------------------------
# c(k) is even and smooth in k; mirroring the first `nmirror` points of the
# midpoint grid (k₁ = Δk/2) to negative k lets the spline see that, instead of
# imposing an artificial end condition at k₁.
function _even_extension(k::AbstractVector, y::AbstractVector, nmirror::Integer)
    return vcat(-reverse(k[1:nmirror]), k), vcat(reverse(y[1:nmirror]), y)
end

"""
    dcf_spline(k_array, c; order=5, smoothing=0.0) -> Dierckx.Spline1D

Spline through the even extension of the tabulated `c(k)`. With `smoothing = 0` it interpolates; a positive `smoothing` (Dierckx's `s`) is meant for noisy tables, e.g. `c = (1 - 1/S)/n` from a measured `S(k)`.
"""
function dcf_spline(k::AbstractVector, c::AbstractVector; order::Integer=5, smoothing::Real=0.0)
    x, y = _even_extension(collect(float.(k)), collect(float.(c)), order + 1)
    return Dierckx.Spline1D(x, y; k=order, s=smoothing)
end

"""
    dcf_derivatives(k_array, c; order=5, smoothing=0.0) -> (c′, c″)
    dcf_derivatives(c::Function, k_array) -> (c′, c″)

First and second derivatives of the direct correlation function on `k_array`: from a quintic spline through the tabulated values `c` (see [`dcf_spline`](@ref)), or by forward-mode differentiation of a callable `c(k)`.

`c″` enters the `q²` vertex of [`compute_gamma`](@ref) directly. A spline through a noisy table differentiates the noise twice, so when `smoothing > 0` is needed, check how `Γ_V` moves with it.
"""
function dcf_derivatives(k::AbstractVector, c::AbstractVector; order::Integer=5, smoothing::Real=0.0)
    smoothing > 0 && @warn "dcf_derivatives: c″ from a smoothing spline is noise-sensitive; it enters Γ_V directly."
    spl = dcf_spline(k, c; order, smoothing)
    return Dierckx.derivative(spl, k; nu=1), Dierckx.derivative(spl, k; nu=2)
end

function dcf_derivatives(c::Function, k::AbstractVector)
    vals = map(x -> _taylor2(c, float(x)), k)
    return getindex.(vals, 2), getindex.(vals, 3)
end

# ---------------------------------------------------------------------------
# Numerical Percus–Yevick for hard hyperspheres in d dimensions.
# ---------------------------------------------------------------------------
# Λ_ν(x) = x^{-ν} J_ν(x), the kernel of the d-dimensional radial Fourier transform (ν = d/2 - 1); Λ_ν′(x) = -x Λ_{ν+1}(x). The power series is used for x < 1, where the closed forms of the half-integer orders cancel.

function _Λ(ν::Real, x::Real)
    if x < 1
        s, term, x2 = zero(x), 1 / gamma(ν + 1), -(x / 2)^2
        for j in 0:25
            s += term
            term *= x2 / ((j + 1) * (ν + j + 1))
        end
        return s / 2^ν
    end
    ν == 0   && return besselj0(x)
    ν == 1   && return besselj1(x) / x
    ν == 1/2 && return sqrt(2 / π) * sin(x) / x
    ν == 3/2 && return sqrt(2 / π) * (sin(x) - x * cos(x)) / x^3
    ν == 5/2 && return sqrt(2 / π) * ((3 - x^2) * sin(x) - 3x * cos(x)) / x^5
    return besselj(ν, x) / x^ν
end

"""
    PYHyperspheres

Converged Percus–Yevick direct correlation function of hard hyperspheres of unit diameter: `c(r)` on the midpoints `r` of the core `[0, 1]` (it vanishes outside). Built by [`solve_PY`](@ref); evaluate its transform with [`dcf`](@ref).
"""
struct PYHyperspheres
    d::Int
    φ::Float64
    ρ::Float64
    Δr::Float64
    r::Vector{Float64}
    c::Vector{Float64}
end

"""
    solve_PY(d, φ; M=400, R=40.0, nsteps=5, tol=1e-12, maxiter=50) -> PYHyperspheres

Percus–Yevick closure for hard hyperspheres of unit diameter in `d` dimensions at packing fraction `φ` (number density `n = φ/v_d`, `v_d` the volume of the unit-diameter sphere).

# Method

With `γ = h - c`, the closure is `c(r) = -1 - γ(r)` inside the core and `c = 0` outside, and Ornstein–Zernike reads `γ̂ = n ĉ²/(1 - n ĉ)`, with the radial transform

    f̂(k) = (2π)^{d/2} ∫dr r^{d-1} Λ_ν(kr) f(r),   Λ_ν(x) = x^{-ν} J_ν(x),  ν = d/2 - 1 .

Only `γ` inside the core enters the closure, so the unknowns are the `M` values of `γ` on the core midpoints `r_n = (n - ½)/M`; the jump of `c` at `r = 1` falls on a cell boundary. The transforms are midpoint sums on `[0, R]` with the conjugate grid `k_m = (m - ½)π/R` (for `d = 3` the pair is the exactly invertible DST-IV), so the discretisation error is `O(1/M²)` — see [`dcf_PY_numeric`](@ref) for the Richardson extrapolation. The equation is solved by Newton's method, with the Jacobian assembled from the transform matrices, along `nsteps` packing fractions up to `φ`.
"""
function solve_PY(d::Integer, φ::Real; M::Integer=400, R::Real=40.0, nsteps::Integer=5,
                  tol::Real=1e-12, maxiter::Integer=50, verbose::Bool=false)
    Δr = 1 / M
    N  = round(Int, R * M)
    Δk = π / (N * Δr)
    r  = Δr .* ((1:M) .- 0.5)
    k  = Δk .* ((1:N) .- 0.5)
    ν  = d / 2 - 1
    vd = volume_d_dim_sphere(d, 1.0)

    # Forward transform of a core-supported function, ĉ = LA c, and the inverse
    # transform evaluated back on the core, γ = BL γ̂.
    L  = [_Λ(ν, k[m] * r[n]) for m in 1:N, n in 1:M]
    LA = L .* ((2π)^(d / 2) * Δr .* r .^ (d - 1))'
    BL = permutedims(L .* ((2π)^(-d / 2) * Δk .* k .^ (d - 1)))

    γ = zeros(M)
    for φs in range(φ / nsteps, φ; length=nsteps)
        ρ = φs / vd
        converged = false
        for it in 1:maxiter
            ĉ   = LA * (-1 .- γ)
            res = γ .- BL * (@. ρ * ĉ^2 / (1 - ρ * ĉ))
            err = norm(res, Inf)
            verbose && println("solve_PY: φ = $φs, iteration $it, residual $err")
            if err < tol
                converged = true
                break
            end
            gp = @. ρ * ĉ * (2 - ρ * ĉ) / (1 - ρ * ĉ)^2
            γ .-= (I + BL * (gp .* LA)) \ res
        end
        converged || error("solve_PY did not converge at φ = $φs (d = $d, M = $M).")
    end
    return PYHyperspheres(d, float(φ), φ / vd, Δr, collect(r), -1 .- γ)
end

"""
    dcf(sol::PYHyperspheres, k) -> (c, c′, c″)

The transform `ĉ(k)` of the core-supported PY direct correlation function and its first two derivatives, by the same midpoint quadrature over the core that the solver uses:

    ĉ′(k) = -(2π)^{d/2} ∫₀¹dr r^{d+1} k Λ_{ν+1}(kr) c(r)
    ĉ″(k) =  (2π)^{d/2} ∫₀¹dr r^{d+1} [k² r² Λ_{ν+2}(kr) - Λ_{ν+1}(kr)] c(r) .
"""
function dcf(sol::PYHyperspheres, k::Real)
    d, ν, Δr = sol.d, sol.d / 2 - 1, sol.Δr
    c0 = c1 = c2 = 0.0
    @inbounds for (rn, cn) in zip(sol.r, sol.c)
        x  = k * rn
        λ1 = _Λ(ν + 1, x)
        c0 += rn^(d - 1) * _Λ(ν, x) * cn
        c1 -= rn^(d + 1) * k * λ1 * cn
        c2 += rn^(d + 1) * (x^2 * _Λ(ν + 2, x) - λ1) * cn
    end
    p = (2π)^(d / 2) * Δr
    return p * c0, p * c1, p * c2
end

"""
    dcf_PY_numeric(d, φ, k_array; M=400, richardson=true, kwargs...) -> (c, c′, c″)

Numerical PY direct correlation function of hard hyperspheres and its first two derivatives on `k_array` ([`solve_PY`](@ref) + [`dcf`](@ref)). With `richardson = true` the `O(1/M²)` discretisation error is removed by combining the solutions for `M` and `M ÷ 2` core points, `(4 X_M - X_{M/2})/3`. `kwargs` are passed to `solve_PY`.
"""
function dcf_PY_numeric(d::Integer, φ::Real, k::AbstractVector; M::Integer=400,
                        richardson::Bool=true, kwargs...)
    table(sol) = (vals = [dcf(sol, x) for x in k];
                  (getindex.(vals, 1), getindex.(vals, 2), getindex.(vals, 3)))
    fine = table(solve_PY(d, φ; M, kwargs...))
    richardson || return fine
    @assert iseven(M) "Richardson extrapolation needs an even M."
    coarse = table(solve_PY(d, φ; M=M ÷ 2, kwargs...))
    return ntuple(i -> (4 .* fine[i] .- coarse[i]) ./ 3, 3)
end

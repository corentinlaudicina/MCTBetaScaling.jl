using LinearAlgebra
using ModeCouplingTheory
import ModeCouplingTheory: dDimModeCouplingKernel

"""
    compute_mu(kernel::dDimModeCouplingKernel, f) -> μ
    compute_mu(kernel::dDimModeCouplingKernel, f, hL) -> μ

Compute the constant

    μ = ∫dk/s_d (h₀ᴸ(k)/k^{d-1}) S(k) c(k) f_c(k) (1 - f_c(k))

of a monodisperse system, with `c(k)` the direct correlation function stored in
the kernel (`n c = 1 - 1/S`).

`f` is the *normalized* critical nonergodicity parameter `f_c(k) = F_c(k)/S(k)` on
`kernel.k_array`. The two-argument form obtains `hL = h₀ᴸ(k)/k^{d-1}` from
[`compute_lambda`](@ref); the three-argument form uses `hL` as given, without
renormalizing it. The integral is taken with the same radial measure
`∫dk/s_d → Σ_k Δk k^{d-1}` that `compute_lambda` pairs `hL` with.
"""
function compute_mu(kernel::dDimModeCouplingKernel, f::AbstractVector)
    _, _, hL, _ = compute_lambda(kernel, f)
    return compute_mu(kernel, f, hL)
end

function compute_mu(kernel::dDimModeCouplingKernel, f::AbstractVector, hL::AbstractVector)
    Nk = kernel.Nk
    @assert length(f) == Nk && length(hL) == Nk "f and hL must live on the kernel grid (kernel.Nk = $Nk)."

    k_array = kernel.k_array
    Δk      = k_array[2] - k_array[1]
    w       = @. Δk * k_array^(kernel.d - 1)

    return sum(@. w * hL * kernel.Sk * kernel.C * f * (1 - f))
end


"""
    S0_PY(η) -> S(0)

Exact `k → 0` limit of the Percus–Yevick hard-sphere structure factor at packing
fraction `η` (compressibility route),

    S(0) = (1 - η)⁴ / (1 + 2η)² .
"""
S0_PY(η::Real) = (1 - η)^4 / (1 + 2η)^2

"""
    extrapolate_S0(kernel::dDimModeCouplingKernel; npoints=3) -> S(0)

Extrapolate the structure factor to `k = 0` from the kernel grid, which starts at
`k₁ = Δk/2`. The direct correlation function `c(k)` stored in the kernel is even
and smooth in `k`, so a polynomial in `k²` is interpolated through its first
`npoints` grid values, `c(0)` is read off as the constant term, and
`S(0) = 1/(1 - n c(0))` (Ornstein–Zernike). No closure is assumed. For PY hard
spheres compare with [`S0_PY`](@ref).
"""
function extrapolate_S0(kernel::dDimModeCouplingKernel; npoints::Integer=3)
    @assert 1 ≤ npoints ≤ kernel.Nk "npoints must lie in 1:kernel.Nk."
    k = kernel.k_array[1:npoints]
    V = [k[i]^(2j) for i in 1:npoints, j in 0:npoints-1]
    c0 = (V \ kernel.C[1:npoints])[1]
    return 1 / (1 - kernel.ρ * c0)
end

"""
    compute_DeltaSigma2(kernel::dDimModeCouplingKernel, f; S0=extrapolate_S0(kernel)) -> Δσ²
    compute_DeltaSigma2(kernel::dDimModeCouplingKernel, f, hL; S0=extrapolate_S0(kernel)) -> Δσ²

Compute the variance of the quenched fluctuations of the separation parameter,

    Δσ² = μ² n_c S(0) ,

with `μ` from [`compute_mu`](@ref) (same arguments, same `hL` convention), `n_c`
the density stored in the kernel and `S0 = S(k→0)`. By default `S(0)` is
extrapolated from the grid with [`extrapolate_S0`](@ref); pass it explicitly to
use an exact value instead, e.g. `S0 = S0_PY(η)` for PY hard spheres.
"""
compute_DeltaSigma2(kernel::dDimModeCouplingKernel, f::AbstractVector;
                    S0::Real=extrapolate_S0(kernel)) =
    compute_mu(kernel, f)^2 * kernel.ρ * S0

compute_DeltaSigma2(kernel::dDimModeCouplingKernel, f::AbstractVector, hL::AbstractVector;
                    S0::Real=extrapolate_S0(kernel)) =
    compute_mu(kernel, f, hL)^2 * kernel.ρ * S0

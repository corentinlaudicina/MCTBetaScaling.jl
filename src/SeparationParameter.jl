using LinearAlgebra
using ModeCouplingTheory
import ModeCouplingTheory: dDimModeCouplingKernel

"""
    compute_C(kernel::dDimModeCouplingKernel, f, dcdn) -> (C, C0)
    compute_C(kernel::dDimModeCouplingKernel, f, dcdn, hL) -> (C, C0)

Compute the constant of the separation parameter `σ = C ε`, `ε = (n - n_c)/n_c`, of a monodisperse system,

    C = ∫dk/s_d (h₀ᴸ(k)/k^{d-1}) C⁽⁰⁾(k) .

`f` is the *normalized* critical nonergodicity parameter `f_c(k) = F_c(k)/S(k)` and `dcdn` the density derivative of the direct correlation function `dc(k)/dn` at `n = n_c`, both on `kernel.k_array`. `dS/dn` is not an input: it follows from Ornstein–Zernike, `(n/S) dS/dn = n S (c + n dc/dn)`.

# Method

`C⁽⁰⁾` is the density derivative of the memory functional at fixed *normalized* `f_c`, the variable in which [`compute_lambda`](@ref) linearizes the plateau equation:

    C⁽⁰⁾(k) = (1-f_c(k))² n_c ∂𝓕_k[f_c]/∂n
            = (1-f_c(k))² [ (1 + (n_c/S(k)) dS(k)/dn) m_c(k)
                            + n_c S(k) ∫dp/(2π)ᵈ ṽᶜ_k(p,k-p) δṽ_k(p,k-p) F_c(p) F_c(|k-p|)
                            + n_c S(k) ∫dp/(2π)ᵈ ṽᶜ_k(p,k-p)² (n_c/S(p)) dS(p)/dn F_c(p) F_c(|k-p|) ] ,

    δṽ_k(p,p') = k⁻¹ k̂·( p n_c dc(p)/dn + p′ n_c dc(p′)/dn ) .

# Returns

`C` and the integrand `C0 = C⁽⁰⁾(k)` on the grid. The four-argument form uses
`hL = h₀ᴸ(k)/k^{d-1}` as given (e.g. from `compute_lambda`), without
renormalizing it.
"""
function compute_C(kernel::dDimModeCouplingKernel, f::AbstractVector, dcdn::AbstractVector)
    _, _, hL, _ = compute_lambda(kernel, f)
    return compute_C(kernel, f, dcdn, hL)
end

function compute_C(kernel::dDimModeCouplingKernel, f::AbstractVector, dcdn::AbstractVector,
                   hL::AbstractVector)
    Nk = kernel.Nk
    @assert length(f) == Nk && length(dcdn) == Nk && length(hL) == Nk "f, dcdn and hL must live on the kernel grid (kernel.Nk = $Nk)."

    d       = kernel.d
    n       = kernel.ρ
    k_array = kernel.k_array
    Sk      = kernel.Sk
    C       = kernel.C
    Δk      = k_array[2] - k_array[1]
    D₀      = kernel.kBT / kernel.m

    fc = collect(float.(f))
    om = @. 1 - fc
    @assert all(x -> x > 0, om) "f_c(k) must be < 1 on the whole grid."

    Cn   = @. n * dcdn                      # n_c dc/dn
    dlnS = @. n * Sk * (C + Cn)             # (n_c/S) dS/dn, Ornstein–Zernike

    # The vertex is symmetric in k ↔ p, so the two legs' dlnS contribute equally.
    C0 = zeros(Float64, Nk)
    @inbounds for iq in 1:Nk
        q  = k_array[iq]
        γq = D₀ * q^2 / Sk[iq]
        sm = 0.0                            # Σ J v²        S f S f
        sx = 0.0                            # Σ J v vₙ      S f S f
        sl = 0.0                            # Σ J v² dlnS_k S f S f
        for ik in 1:Nk
            k = k_array[ik]
            for ip in abs(iq - ik) + 1 : min(Nk, ik + iq - 1)
                p  = k_array[ip]
                a  = q^2 + k^2 - p^2
                b  = q^2 - k^2 + p^2
                v  = a * C[ik]  + b * C[ip]
                vn = a * Cn[ik] + b * Cn[ip]
                w  = kernel.J[iq, ik, ip] * Sk[ik] * fc[ik] * Sk[ip] * fc[ip]
                sm += w * v^2
                sx += w * v * vn
                sl += w * v^2 * dlnS[ik]
            end
        end
        m_c = kernel.prefactor * sm / γq
        C0[iq] = om[iq]^2 * ((1 + dlnS[iq]) * m_c + 2 * kernel.prefactor * (sx + sl) / γq)
    end

    # ∫dk/s_d (h₀ᴸ/k^{d-1}) g(k) → Σ_k Δk k^{d-1} hL_k g_k, as in compute_lambda.
    w = @. Δk * k_array^(d - 1)
    return (sum(@. w * hL * C0), C0)
end

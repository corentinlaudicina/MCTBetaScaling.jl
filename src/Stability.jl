using LinearAlgebra

"""
    stability_matrix(kernel::BipolarMCTKernel, F) -> C0

MCT stability matrix in the `δF` representation at the plateau `F` (unnormalised, `F = S f`),

    C0[i,j] = ρ A(k_i)² Σ_l w_{ijl} U₀(k_i,p_j,r_l)² B(r_l),   A = S - F,  B = F ,

the exact Jacobian `∂Φ_i/∂F_j` of the discrete plateau map [`steady_state_map`](@ref). It has non-negative entries, so its top eigenvalue `E0` is real and simple (Perron); `E0 = 1` at the critical point. The `f`-representation used by [`compute_lambda`](@ref) is `C1 = S⁻¹ C0 S`.
"""
function stability_matrix(K::BipolarMCTKernel{T}, F::AbstractVector) where {T}
    Nk, k, C = K.Nk, K.k_array, K.C
    A  = K.Sk .- F
    C0 = zeros(promote_type(T, eltype(F)), Nk, Nk)
    Threads.@threads for i in 1:Nk
        buf = Vector{T}(undef, Nk)
        k2  = k[i]^2
        pre = K.ρ * A[i]^2 / k2^2                   # U₀² = N₀²/k⁴
        for j in 1:Nk
            lo, hi = triangle_bounds(i, j, Nk)
            w  = triangle_weights(K, buf, i, j, lo, hi)
            p2 = k[j]^2
            cp = C[j]
            s  = zero(eltype(C0))
            @inbounds @simd for l in lo:hi
                N0 = _N0(k2, p2, k[l]^2, C[l], cp)
                s += w[l-lo+1] * N0^2 * F[l]
            end
            C0[i, j] = pre * s
        end
    end
    return C0
end

"""
    perron_pair(C) -> (E0, ρR, ℓ)

Largest real eigenvalue of `C` with its right (`C ρR = E0 ρR`) and left (`ℓᵀ C = E0 ℓᵀ`) eigenvectors, both scaled to unit sum (positive for a Perron matrix).
"""
function perron_pair(C::AbstractMatrix)
    ER = eigen(C)
    iR = argmax(real.(ER.values))
    ρR = real.(ER.vectors[:, iR])
    EL = eigen(collect(transpose(C)))
    ℓ  = real.(EL.vectors[:, argmax(real.(EL.values))])
    ρR ./= sum(ρR)
    ℓ  ./= sum(ℓ)
    return real(ER.values[iR]), ρR, ℓ
end

"""
    compute_lambda(kernel::BipolarMCTKernel, f) -> (λ, hR, hL, C1)

[`compute_lambda`](@ref) for a [`BipolarMCTKernel`](@ref): same conventions and returns (normalised `f`, Götze normalisations, `hL = h₀ᴸ(k)/k^{d-1}`, `C1 = C⁽¹⁾` in the `f`-representation), built on the kernel's own weights.

Both normalisations divide out of the ratio

    λ = Σ_i ĥ_i (1-f_i)² m_i[S∘h] / Σ_i ĥ_i h_i²/(1-f_i),   h = ρR/S,  ĥ = S∘ℓ ,

with `ρR`, `ℓ` the Perron vectors of [`stability_matrix`](@ref) and `m` the [`memory_functional`](@ref).
"""
function compute_lambda(K::BipolarMCTKernel, f::AbstractVector)
    Nk = K.Nk
    @assert length(f) == Nk "length(f) must match the kernel grid (kernel.Nk = $Nk)."
    d, k, S, Δk = K.d, K.k_array, K.Sk, K.Δk
    fc = collect(float.(f))
    om = @. 1 - fc
    @assert all(x -> x > 0, om) "f_c(k) must be < 1 on the whole grid."

    C0 = stability_matrix(K, S .* fc)
    _, ρR, ℓ = perron_pair(C0)
    C1 = Diagonal(1 ./ S) * C0 * Diagonal(S)

    hR = ρR ./ S                             # right eigenvector of C1
    hL = @. S * ℓ / k^(d - 1)                # (left eigenvector of C1)/k^{d-1}
    w  = @. Δk * k^(d - 1)
    Σ1 = sum(@. w * hL * hR)
    Σ2 = sum(@. w * hL * hR^2 / om)
    hR .*= Σ1 / Σ2
    hL .*= Σ2 / Σ1^2

    mh = memory_functional(K, S .* hR)      # Σ_{k,p} A[q,k,p] hR_k hR_p
    λ  = sum(@. w * hL * om^2 * mh)
    return (λ, hR, hL, C1)
end

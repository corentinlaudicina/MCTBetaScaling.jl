
using LinearAlgebra
using ModeCouplingTheory



"""
    compute_lambda(V, f; check_symmetry=true)

Compute the MCT exponent parameter λ from the vertex `V[q,k,p]`
(including weights and prefactor, symmetric in k↔p) and nonergodicity parameters `f[q]`.

Eqs used: C_{qk}=∑ₚ V[q,k,p] f[p](1-f[k])²;  C_{q,kp}=½ V[q,k,p](1-f[k])²(1-f[p])²;
λ=∑_{q,k,p} ℓ[q] C_{q,kp} r[k] r[p].
Normalizations imposed: ∑ₖ ℓₖ rₖ = 1 and ∑ₖ rₖ² ℓₖ (1-fₖ) = 1.
"""
function compute_lambda(V::Array{<:Real,3}, f::AbstractVector; check_symmetry::Bool=true)
    N1,N2,N3 = size(V)
    @assert length(f) == N1 == N2 == N3 "V must be NxNxN and match length(f)."
    N = length(f)
    f = collect(float.(f))
    V = Array{Float64,3}(V)
    if check_symmetry
        for q in 1:N, k in 1:N, p in 1:N
            @assert isapprox(V[q,k,p], V[q,p,k]; atol=1e-12, rtol=1e-12) "V must be symmetric in k↔p"
        end
    end

    one_minus_f_sq = @. (1 - f)^2

    # C_{qk} = ∑_p V[q,k,p] f[p] (1-f[k])^2
    C = zeros(Float64, N, N)
    for q in 1:N, k in 1:N
        s = 0.0
        for p in 1:N
            s += V[q,k,p] * f[p]
        end
        C[q,k] = 2* s * one_minus_f_sq[k]
    end

    # leading right/left eigenvectors (eigval ≈ 1 at criticality)
    eigR = eigen(C)
    idx  = argmax(real.(eigR.values))
    r0   = real.(eigR.vectors[:, idx])
    eigL = eigen(transpose(C))
    idxL = argmax(real.(eigL.values))
    l0   = real.(eigL.vectors[:, idxL])

    # Enforce the two normalizations:
    S = dot(l0, r0)
    T = sum((r0 .^ 2) .* l0 .* (1 .- f))
    @assert S != 0 && T != 0 "Degenerate eigenvectors (S or T = 0)."
    a = S / T
    b = T / S^2
    r = a .* r0
    ℓ = b .* l0

    # λ = ½ ∑_{q,k,p} ℓ[q] V[q,k,p] (1-f[k])^2 (1-f[p])^2 r[k] r[p]
    λ = 0.0
    for q in 1:N
        s_q = 0.0
        for k in 1:N
            rk = r[k] * one_minus_f_sq[k]
            t = 0.0
            for p in 1:N
                t += V[q,k,p] * (one_minus_f_sq[p] * r[p])
            end
            s_q += rk * t
        end
        λ += ℓ[q] * s_q #/ (one_minus_f_sq[q])
    end

    return (λ, r, ℓ, C)
end


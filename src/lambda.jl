
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
    @inbounds for q in 1:N, k in 1:N
        s = 0.0
        @simd for p in 1:N
            s += V[q,k,p] * f[p]
        end
        C[q,k] = s * one_minus_f_sq[k]
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
    @inbounds for q in 1:N
        s_q = 0.0
        for k in 1:N
            rk = r[k] * one_minus_f_sq[k]
            t = 0.0
            @simd for p in 1:N
                t += V[q,k,p] * (one_minus_f_sq[p] * r[p])
            end
            s_q += rk * t
        end
        λ += ℓ[q] * s_q
    end
    λ *= 0.5

    return (λ, r, ℓ, C)
end

# """
#     find_analytical_C_k(k, η)
# Finds the direct correlation function given by the 
# analytical Percus-Yevick solution of the Ornstein-Zernike 
# equation for hard spheres for a given volume fraction η.

# Reference: Wertheim, M. S. "Exact solution of the Percus-Yevick integral equation 
# for hard spheres." Physical Review Letters 10.8 (1963): 321.
# """ 
# function find_analytical_C_k(k, η)
#     A = -(1 - η)^-4 *(1 + 2η)^2
#     B = (1 - η)^-4*  6η*(1 + η/2)^2
#     D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
#     Cₖ = @. 4π/k^6 * 
#     (
#         24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k)
#      + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k)
#      )
#     return Cₖ
# end

# """
#     find_analytical_S_k(k, η)
# Finds the static structure factor given by the 
# analytical Percus-Yevick solution of the Ornstein-Zernike 
# equation for hard spheres for a given volume fraction η.
# """ 
# function find_analytical_S_k(k, η)
#         Cₖ = find_analytical_C_k(k, η)
#         ρ = 6/π * η
#         Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
#     return Sₖ
# end

# # We solve MCT for hard spheres at a volume fraction of 0.51591
# η = 0.51593; ρ = η*6/π; kBT = 1.0; m = 1.0

# Nk = 100; kmax = 40.0; 
# dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5) # construct the grid this way to satisfy the assumptions
#                                                   # of the discretization.
# Sₖ = find_analytical_S_k(k_array, η)

# ∂F0 = zeros(Nk); α = 1.0; β = 0.0; γ = @. k_array^2*kBT/(m*Sₖ); δ = 0.0

# kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)
# sol = solve_steady_state(γ, Sₖ, kernel; tolerance=10^-8, verbose=false)
# fk = get_F(sol, 1, :)

# V = kernel.prefactor .* kernel.V .* kernel.J

# λ, r, l, C = compute_lambda(V, fk; check_symmetry=true)



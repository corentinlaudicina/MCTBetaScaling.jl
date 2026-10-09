using LinearAlgebra
using ModeCouplingTheory
import ModeCouplingTheory: dDimModeCouplingKernel, MultiComponentModeCouplingKernel3D,
                          evaluate_kernel
const _SMatrix = ModeCouplingTheory.SMatrix   # re-exported from StaticArrays by the package

"""
    compute_lambda(kernel::dDimModeCouplingKernel, f) -> (λ, hR, hL, C1)

Compute the mode-coupling exponent parameter `λ` of the mean-field (one-loop) MCT scenario, following the conventions of the asymptotic analysis around the ergodic–nonergodic transition.

`f` is the *normalized* critical nonergodicity parameter
`f_c(k) = F_c(k) / S(k) ∈ [0, 1]`, given on the kernel grid `kernel.k_array`.

# Method

With `γ_q = (kBT/m) q² / S(q)` the bare relaxation rate, the normalized memory functional at the plateau reads `F_c(q)/(1-F_c(q)) ≡ 𝓕_q[f] = Σ_{k,p} A[q,k,p] f_k f_p`, where `A[q,k,p]` is the symmetric mode-coupling vertex assembled from the kernel (prefactor, angular weight `J`, vertex `V`, and structure factors).

The stability operator (Jacobian of the memory kernel) is

    C⁽¹⁾[q,k] = (1-f_c(q))² ∂𝓕_q/∂f_k = 2 (1-f_c(q))² Σ_p A[q,k,p] f_p          (Eq. MCT_stab_matrix_3D)

Its largest eigenvalue equals `1` at the critical point. Let `h₀ᴿ` be the corresponding right eigenvector and `h₀ᴸ(k)/k^{d-1}` the corresponding left eigenvector,

    Σ_p C⁽¹⁾[k,p] h₀ᴿ(p)                 = E₀ h₀ᴿ(k)
    Σ_p (h₀ᴸ(p)/p^{d-1}) C⁽¹⁾[p,k]       = (h₀ᴸ(k)/k^{d-1}) E₀ ,

fixed by the two Götze normalizations (with `∫dk/s_d → Σ_k Δk k^{d-1}`)

    ∫dk/s_d (h₀ᴸ(k)/k^{d-1}) h₀ᴿ(k)              = 1                          
    ∫dk/s_d (h₀ᴸ(k)/k^{d-1}) h₀ᴿ(k)²/(1-f_c(k))  = 1 .                        

The exponent parameter is then

    λ = ∫dk/s_d ∫dp/(2π)ᵈ (h₀ᴸ(k)/k^{d-1}) C⁽²⁾(k,p) h₀ᴿ(|k-p|) h₀ᴿ(p)

with the second-variation of memory kernel
    C⁽²⁾[q,k,p] = ½ n S(q) (1-f_c(q))² ṽ_q(k,p)² S(k) S(p) = (1-f_c(q))² A[q,k,p] .

# Returns

`λ`, the normalized right eigenvector `hR = h₀ᴿ`, the normalized left eigenvector `hL = h₀ᴸ(k)/k^{d-1}`, and the stability matrix `C1 = C⁽¹⁾`.
"""
compute_lambda(kernel::dDimModeCouplingKernel, f::AbstractVector) =
    compute_lambda(BipolarMCTKernel(kernel), f)


"""
    compute_lambda_multicomponent(kernel::MultiComponentModeCouplingKernel3D, f::AbstractArray)
        -> (λ, hR, hL, C1)

`N`-species generalization of [`compute_lambda`](@ref).  Computes the mode-coupling exponent parameter `λ` for a mixture whose critical nonergodicity parameter is the *un-normalized*, symmetric `N×N` matrix field `f = f_cᵅᵝ(k) = F_cᵅᵝ(k)` given on `kernel.k_array` (`0 ⪯ f_c(k) ⪯ S(k)` in the matrix sense).

`f` may be supplied either as a length-`Nk` `AbstractVector` of `N×N` matrices (the layout the package uses for `F(k,t)` and for the nonergodicity parameter returned by `get_F`), or as an `Nk×N×N` array; `hR`, `hL` come back in the same layout.

# Method

Every scalar object of the one-component analysis is promoted to the space of symmetric `N×N` matrices attached to each `k`-point (`N(N+1)/2` independent components per point).

## Bilinear memory functional

The package already precomputes the mode-coupling vertex/prefactor tensors and contracts them: `evaluate_kernel(kernel, F, 0)` returns the matrix field `Kᵅᵝ(k) = 𝓜ᵅᵝ[F,F](k)`, an exactly homogeneous quadratic form in `F`.  The symmetric bilinear functional is therefore recovered *without re-deriving the vertex* by polarization

    M[A,B](k) = ¼ ( 𝓜[A+B, A+B](k) − 𝓜[A−B, A−B](k) )    (Eq.MCT_EOS_N)

with `M[F,F] = 𝓜[F,F] = evaluate_kernel(kernel, F, 0)`.

## Stability super-operator

The package solves the plateau (steady-state) equation as the fixed point `γ(k) f_c(k) = K_c(k) (S(k) − f_c(k))` with `γ(k) = kBT k² Diagonal(x ./ m) S(k)⁻¹`, `x = ρ / Σρ`.  Linearizing it and symmetrizing the two equivalent forms of the result gives the stability map on symmetric-matrix fields

    (C⁽¹⁾·δf)ᵅᵝ(q) = 1/(2 kBT q²) Σ_γδ [S(q)−f_c(q)]ᵅᵧ
                       { Wᵧᵧ Gᵧδ(q) + Gᵧδ(q) Wδδ } [S(q)−f_c(q)]δᵝ        (Eq. MCT_stab_N)

    G(q) = M[f_c, δf](q) + M[δf, f_c](q) ,   W = Diagonal(m ./ x) .

`C⁽¹⁾` is built as a dense `[Nk·N(N+1)/2]²` matrix in an orthonormal basis of the symmetric-matrix space (diagonal units `Eᵅᵅ`, off-diagonal units `(Eᵅᵝ+Eᵝᵅ)/√2`), so that the Euclidean inner product of coefficient vectors equals the trace inner product `⟨A|B⟩ = Σ_k w_k Tr[A(k)B(k)]`, `w_k = Δk k^{d-1}`.  Its largest eigenvalue `E0` equals `1` at criticality; `hR = e_c` is the leading right eigen-field and `U` the leading left eigen-field, with `hL(k) = U(k) / k^{d-1}` (as in the scalar code).

Complexity: `O(Nk³ · N(N+1)/2 · N⁴)` to fill `C⁽¹⁾` (two `evaluate_kernel` calls per column) plus `O((Nk·N(N+1)/2)³)` for the dense diagonalization.  Not optimized.

## Normalizations and λ

Using the trace inner product (off-diagonal species pairs are double-counted by the trace, which is the correct combinatorial weight):

    ⟨hL | hR⟩                                   = 1
    ⟨hL | hR(q) [S(q)−f_c(q)]⁻¹ hR(q)⟩          = 1                       (Eq. norm_N)

the second is the matrix lift of the scalar `hR²/(1−f_c)` (see notes below).  With `hR`, `hL` fixed by these two conditions,

    λ = ⟨hL | 1/(2 kBT q²) [S(q)−f_c(q)] { W M[hR,hR](q) + M[hR,hR](q) W } [S(q)−f_c(q)]⟩
        (Eq. lambda_N)

# Returns

`λ`, the right eigen-field `hR`, the left eigen-field `hL = U/k^{d-1}` (both normalized by `Eq. norm_N` and in the layout of `f`), and the dense stability matrix `C1`.

For `N=1` this reproduces `compute_lambda` (same `λ`, same `E0`, `hR` equal up to the normalization convention) — see `test/test_lambda_multicomponent.jl`.
"""
function compute_lambda_multicomponent(kernel::MultiComponentModeCouplingKernel3D,
                                       f::AbstractArray)
    Nk = kernel.Nk
    Ns = kernel.Ns
    d  = 3                                  # the multi-component 3D kernel is fixed to d = 3
    k_array = collect(float.(kernel.k_array))
    Δk = k_array[2] - k_array[1]
    w  = @. Δk * k_array^(d - 1)
    kBT   = kernel.kBT
    x     = kernel.ρ ./ sum(kernel.ρ)
    ρ_all = sum(kernel.ρ)
    Wvec  = kernel.m ./ x                   # W = Diagonal(m ./ x)

    # --- input normalization: accept Vector{<:AbstractMatrix} or Nk×Ns×Ns array ---
    as_vec_of_mat(A::AbstractVector) = [Matrix{Float64}(A[i]) for i in eachindex(A)]
    as_vec_of_mat(A::AbstractArray{<:Real,3}) = [Matrix{Float64}(A[i, :, :]) for i in 1:size(A, 1)]
    fc = as_vec_of_mat(f)
    @assert length(fc) == Nk "f must have Nk = $Nk entries (one symmetric $Ns×$Ns matrix per k-point)."
    @assert all(F -> size(F) == (Ns, Ns), fc) "each f[k] must be $Ns×$Ns."
    @assert all(F -> norm(F - F') <= 1e-8 * max(norm(F), 1), fc) "each f[k] must be symmetric."
    fc = [Symmetric(0.5 * (F + F')) |> Matrix for F in fc]

    # --- recover S(k) from the kernel's stored direct correlation function ---
    #     kernel.C[i] = (Diagonal(1 ./ x) - S(k)⁻¹) / ρ_all   (see MultiComponentModeCouplingKernel)
    Sk = [Matrix(Symmetric(inv(Diagonal(1 ./ x) .- ρ_all .* Matrix(kernel.C[i])))) for i in 1:Nk]
    Nq = [Matrix(Symmetric(Sk[i] .- fc[i])) for i in 1:Nk]          # S(q) - f_c(q)

    # --- bilinear memory functional through the package's own vertex contraction ---
    _tosm(A) = Vector{_SMatrix{Ns,Ns,Float64}}([_SMatrix{Ns,Ns,Float64}(A[i]) for i in 1:Nk])
    Qdiag(F) = evaluate_kernel(kernel, _tosm(F), 0.0).diag           # 𝓜[F,F](k), homogeneous deg 2
    function Mbil(A, B)                                              # symmetric bilinear M[A,B], Eq. MCT_EOS_N
        Qp = Qdiag([A[i] .+ B[i] for i in 1:Nk])
        Qm = Qdiag([A[i] .- B[i] for i in 1:Nk])
        [(Matrix(Qp[i]) .- Matrix(Qm[i])) ./ 4 for i in 1:Nk]
    end

    # --- orthonormal basis of the symmetric Ns×Ns matrices (trace inner product) ---
    basis = Matrix{Float64}[]
    for a in 1:Ns
        E = zeros(Ns, Ns); E[a, a] = 1.0; push!(basis, E)
    end
    for a in 1:Ns, b in (a + 1):Ns
        E = zeros(Ns, Ns); E[a, b] = E[b, a] = 1 / sqrt(2); push!(basis, E)
    end
    Np   = length(basis)                                            # N(N+1)/2
    Ndof = Nk * Np
    lin(ik, a) = (ik - 1) * Np + a

    # --- assemble the dense stability super-operator, Eq. MCT_stab_N ---------------
    # column (ik0, b): unit perturbation basis[b] placed at the single k-point ik0.
    C1 = zeros(Ndof, Ndof)
    Wd = Diagonal(Wvec)
    for ik0 in 1:Nk, b in 1:Np
        δf = [zeros(Ns, Ns) for _ in 1:Nk]
        δf[ik0] = copy(basis[b])
        Gh = Mbil(fc, δf)                                           # M[f_c, δf]
        col = lin(ik0, b)
        for iq in 1:Nk
            G   = 2 .* Gh[iq]                                       # M[f_c,δf] + M[δf,f_c]
            mid = Wd * G .+ G * Wd
            resp = (1 / (2 * kBT * k_array[iq]^2)) .* (Nq[iq] * mid * Nq[iq])
            resp = 0.5 .* (resp .+ resp')
            for a in 1:Np
                C1[lin(iq, a), col] = tr(basis[a] * resp)
            end
        end
    end

    # --- leading right / left eigen-fields (E0 → 1 at criticality) -----------------
    ER = eigen(C1)
    iR = argmax(real.(ER.values))
    vR = real.(ER.vectors[:, iR])
    EL = eigen(collect(transpose(C1)))
    iL = argmax(real.(EL.values))
    vL = real.(EL.vectors[:, iL])

    field(v) = [sum(v[lin(ik, a)] .* basis[a] for a in 1:Np) for ik in 1:Nk]
    hR = field(vR)
    U  = field(vL)
    hL = [U[ik] ./ k_array[ik]^(d - 1) for ik in 1:Nk]              # h₀ᴸ(k) / k^{d-1}

    inner(A, B) = sum(w[ik] * tr(A[ik] * B[ik]) for ik in 1:Nk)     # ⟨A|B⟩

    # --- fix the two Götze normalizations, Eq. norm_N -----------------------------
    Ninv = [inv(Nq[ik]) for ik in 1:Nk]
    Nfun(h) = [(P = h[ik] * Ninv[ik] * h[ik]; 0.5 .* (P .+ P')) for ik in 1:Nk]
    Σ1 = inner(hL, hR)
    Σ2 = inner(hL, Nfun(hR))
    @assert Σ1 != 0 && Σ2 != 0 "Degenerate eigen-fields (Σ1 or Σ2 = 0)."
    hR = [(Σ1 / Σ2) .* hR[ik] for ik in 1:Nk]
    hL = [(Σ2 / Σ1^2) .* hL[ik] for ik in 1:Nk]

    # --- exponent parameter, Eq. lambda_N ---------------------------------------
    # Same W/kBT weighting as C⁽¹⁾: λ is the second-order term of the same map.
    MhR = Mbil(hR, hR)                                              # = 𝓜[hR,hR]
    λterm = [(1 / (2 * kBT * k_array[ik]^2)) .* (Nq[ik] * (Wd * MhR[ik] .+ MhR[ik] * Wd) * Nq[ik])
             for ik in 1:Nk]
    λ = inner(hL, λterm)

    # --- return hR, hL in the layout of the input ------------------------------
    if f isa AbstractArray{<:Real,3}
        hRo = zeros(Nk, Ns, Ns); hLo = zeros(Nk, Ns, Ns)
        for ik in 1:Nk
            hRo[ik, :, :] .= hR[ik]; hLo[ik, :, :] .= hL[ik]
        end
        return (λ, hRo, hLo, C1)
    elseif eltype(f) <: _SMatrix
        return (λ, [_SMatrix{Ns,Ns,Float64}(hR[ik]) for ik in 1:Nk],
                   [_SMatrix{Ns,Ns,Float64}(hL[ik]) for ik in 1:Nk], C1)
    else
        return (λ, hR, hL, C1)
    end
end

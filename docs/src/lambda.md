# Exponent Parameter $\lambda$, Eigenvectors, and Stability Matrix

The function `compute_lambda` evaluates the mode-coupling theory (MCT) exponent parameter $\lambda$ together with the leading right and left eigenvectors and the stability matrix $C$.

In MCT, the glass instability is characterized by the largest eigenvalue of the stability matrix approaching unity; $\lambda$ then fixes the critical exponents via standard transcendental relations (see Götze 1985).

## Function

```compute_lambda(V::Array{<:Real,3}, f::AbstractVector; check_symmetry::Bool=true)```

## Inputs

`V[k,q,p]` — 3-tensor of mode-coupling vertices. Must be symmetric in the last two indices (q ↔ p), must be normalized such that the discretized integral term in the equations is given by `\Omega_k \sum_{qp} V_{k,q,p} f_q, f_p S_q S_p` (note the additional factors `\Omega_k`, `S_q`, and `S_p`). This ensures that the equation for `f_k` is given by $\frac{f_k}{1-f_k}=M_q[f]$. 

`f[k]` — nonergodicity parameters `f_k` on the same grid as V. It must be normalized with respect to the structure factor.

`check_symmetry` — when true, asserts the (q,p) symmetry.

## Returns

A tuple `(λ, r, l, C)`:

`λ::Float64` — the MCT exponent parameter.

`r::Vector{Float64}`  — normalized right eigenvector of the dominant eigenvalue of C.

`l::Vector{Float64}` — normalized left eigenvector.

`C::Matrix{Float64}` — stability matrix

$$C_{qk} = 2 (1-f_q)^2 \sum_p V{kqp} f_p$$

## Normalizations

The eigenvectors are rescaled such that

$$\sum_q l_k r_k = 1, $$

$$\sum_q (1-f_k) l_k r_k^2 = 1.$$

### Exponent parameter λ

The exponent parameter is computed using
$$\lambda = \frac{1}{2} \sum_{k,q,p} l_k V_{kqp} (1-f_q)^2 (1-f_p)^2 r_q r_p.$$


## Example (hard spheres)

The code below shows how to compute the value of $\lambda$ and the associated eigenvectors, with respect to a kernel implemented in `ModeCouplingTheory.jl`. Note that in the convention of that package, the kernel includes the prefactor $\Omega(k)$, and the non-ergodicity parameter is not normalized with respect to $S(k)$

```julia
using MCTBetaScaling, ModeCouplingTheory


# PY Structure factor and C(k):
function find_analytical_C_k(k, η)
    A = -(1 - η)^-4 *(1 + 2η)^2
    B = (1 - η)^-4*  6η*(1 + η/2)^2
    D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
    Cₖ = @. 4π/k^6 * 
    (
        24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k)
    + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k)
    )
    return Cₖ
end
function find_analytical_S_k(k, η)
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end


η = 0.51593
ρ = η * 6/π
kBT = 1.0
m = 1.0

Nk = 100
kmax = 40.0
dk = kmax / Nk
k_array = dk .* (collect(1:Nk) .- 0.5) # midpoint grid

Sₖ = find_analytical_S_k(k_array, η)
Ωₖ = @. k_array^2 * kBT / (m * Sₖ)

kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, kBT, m, k_array, Sₖ, 3)

# find the nonergodicity-parameter
sol = solve_steady_state(Ωₖ, Sₖ, kernel; tolerance=1e-8, verbose=false)
Fk = get_F(sol, 1, :)

# in principle the vertex should be constructre by the user.
# here, the implemenation is 
# K[q] = kernel.prefactor[q] * sum_kp kernel.V[q,k,p] * kernel.J[q,k,p] * F[k] * F[p]
# So we can construct the vertex directly from the kernel as follows:
V = kernel.prefactor .* kernel.V .* kernel.J
for q in 1:Nk, k in 1:Nk, p in 1:Nk
    V[q,k,p] = V[q,k,p] * Sₖ[k] * Sₖ[p] / Ωₖ[q]
end

# Normalize fk
fk = Fk ./ Sₖ

# Compute λ
λ, r, l, C = compute_lambda(V, fk; check_symmetry=true)
@show λ
```

which prints `λ = 0.7222348083592546` 

## References

W. Götze, Properties of the Glass Instability Treated within a Mode Coupling Theory,
Z. Phys. B 60, 195–203 (1985).

U. Bengtzelius, W. Götze, A. Sjölander, Dynamics of supercooled liquids and the glass transition,
J. Phys. C 17, 5915 (1984).
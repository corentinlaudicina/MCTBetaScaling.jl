#
# Variance of the quenched fluctuations of the separation parameter,
#
# Δσ² = μ² n_c S(0), with
# μ = ∫dk/s_d (h₀ᴸ(k)/k^{d-1})S(k)c(k)f_c(k)(1 - f_c(k)) ,
# for Percus–Yevick hard spheres.
#
import Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
using ModeCouplingTheory, MCTBetaScaling

# --- Wavevector grid -------------------------------------------------------
Nk   = 300
kmax = 60.0
dk   = kmax / Nk
k    = dk .* (collect(1:Nk) .- 0.5)

η  = 0.51593
ρ  = 6 / π * η
Sk = S_PY.(k, η)
kernel = ModeCouplingTheory.dDimModeCouplingKernel(ρ, 1.0, 1.0, k, Sk, 3)
sol    = solve_steady_state(@.(k^2 / Sk), Sk, kernel; tolerance=1e-8)
fc     = get_F(sol, 1, :) ./ Sk

println("at η = $η:")
println("  μ                                        : ", round(compute_mu(kernel, fc); sigdigits=5))

println("  S(0): extrapolated / exact PY            : ", round(extrapolate_S0(kernel); sigdigits=6),
        " / ", round(S0_PY(η); sigdigits=6))
println("  Δσ² with the extrapolated S(0)           : ", round(compute_DeltaSigma2(kernel, fc); sigdigits=5))
println("  Δσ² with the exact PY S(0)               : ", round(compute_DeltaSigma2(kernel, fc; S0=S0_PY(η)); sigdigits=5))

# --- Critical value, ε → 0 -----------------------------------------------------
# μ is a small remainder of a sign-changing integrand, and η = 0.51593 sits ε ≈ 3.5e-5 above this grid's critical point, where μ still moves like √ε (by ~20%, so Δσ² by ~40%). So locate φc of the discrete equation, solve the plateau just above it, and fit μ = μc + a√ε + bε.

template = BipolarMCTKernel(1.0, 1.0, 1.0, k, ones(Nk), 3)
system(φ) = BipolarMCTKernel(template, 6φ / π, S_PY.(k, φ))
φc = find_critical_packing(system, 0.52).φc

εs = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
μs = Float64[]
F  = nothing
for ε in εs
    K = system(φc * (1 + ε))
    global F = steady_state(K; F0 = F === nothing ? K.Sk : F)
    kd = ModeCouplingTheory.dDimModeCouplingKernel(K.ρ, 1.0, 1.0, k, K.Sk, 3)
    push!(μs, compute_mu(kd, F ./ K.Sk))
end
μc   = ([ones(length(εs)) sqrt.(εs) εs] \ μs)[1]
nc   = 6φc / π
Δσ²c = μc^2 * nc * S0_PY(φc)

println("\nε → 0 (φc = $(round(φc; digits=10))):")
for (ε, μ) in zip(εs, μs)
    println("  ε = $(rpad(ε, 7))  μ = ", round(μ; sigdigits=5))
end
println("  μc                                       : ", round(μc; sigdigits=5))
println("  Δσ²c = μc² n_c S(0)                      : ", round(Δσ²c; sigdigits=5))

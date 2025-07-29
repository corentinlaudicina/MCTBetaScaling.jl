import ModeCouplingTheory.allocate_temporary_arrays
import ModeCouplingTheory.initialize_F_temp!
import ModeCouplingTheory.initialize_integrals!
import ModeCouplingTheory.update_Fuchs_parameters!
import ModeCouplingTheory.update_F!
import ModeCouplingTheory.AbstractNoKernelEquation

using SpecialFunctions
include("HelperFunctions.jl")
 
# mutable struct StochasticBetaScalingEquationCoefficients{T}
#     λ::T
#     σ::T
#     t₀::T
#     δ::T
#     δ_times_t::T
#     a::T
#     b::T
# end

mutable struct StochasticBetaScalingEquationCoefficients{T}
    λ::T                    
    σ::Vector{T}            # spatially varying
    t₀::T
    δ::T                   
    δ_times_t::Vector{T}    # must match σ
    a::T
    b::T
end

# struct StochasticBetaScalingEquation{T,A,B,C,D} <: AbstractNoKernelEquation
#     coeffs::T
#     F₀::A
#     K₀::B
#     kernel::C
#     update_coefficients!::D
# end

struct StochasticBetaScalingEquation{T,A,B,C,D} <: AbstractNoKernelEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
    Nx::Int ## number of spatial sites
end

# function Base.show(io::IO, ::MIME"text/plain", p::StochasticBetaScalingEquation)
#     println(io, "MCT beta-scaling object:")
#     println(io, "   σ - δ t + λ (g(t))² = ∂ₜ∫g(t-τ)g(τ)dτ")
#     println(io, "with real-valued parameters.")
# end

function Base.show(io::IO, ::MIME"text/plain", p::StochasticBetaScalingEquation)
    println(io, "MCT β-scaling object with spatial structure:")
    println(io, "   σ(x) - δ t + λ (g(x,t))² = ∂ₜ∫g(x,t-τ)g(x,τ)dτ  for each x")
    println(io, "with real-valued, spatially varying parameters σ[x].")
end

# function StochasticBetaScalingEquationCoefficients(λ, σ, t₀, δ)
#     exponent_func = (x) -> (SpecialFunctions.gamma(1 - x) / SpecialFunctions.gamma(1 - 2x)) * SpecialFunctions.gamma(1 - x) - λ
#     a = regula_falsi(0.2, 0.3, exponent_func)
#     b = -regula_falsi(-0.5, -0.1, exponent_func)
#     return StochasticBetaScalingEquationCoefficients(λ, σ, t₀, δ, Float64(0.0), a, b)
# end

function StochasticBetaScalingEquationCoefficients(λ, σ_vec::Vector{Float64}, t₀, δ)
    exponent_func = (x) -> (gamma(1 - x) / gamma(1 - 2x)) * gamma(1 - x) - λ
    a = regula_falsi(0.2, 0.3, exponent_func)
    b = -regula_falsi(-0.5, -0.1, exponent_func)
    δ_times_t = zeros(Float64, length(σ_vec))
    return StochasticBetaScalingEquationCoefficients(λ, σ_vec, t₀, δ, δ_times_t, a, b)
end

# """
#     BetaScalingEquation(λ, σ, t₀, δ=0.0)

# Defines the β-scaling equation of MCT, which is a scalar equation for
# a single scaling function g(t), determined by scalar parameters
# λ (the MCT exponent parameter, 1/2<=λ<1), the distance parameter to
# the glass transition σ, and (for convenience) an arbitrary time scale
# t₀ that just shifts the results. Optionally, a "hopping parameter"
# δ can be given (defaults to zero).

# The MCT exponents a and b will be automatically calculated from λ.
# """
# function StochasticBetaScalingEquation(λ::Float64, σ::Float64, t₀::Float64; δ=0.0)
#     StochasticBetaScalingEquation(StochasticBetaScalingEquationCoefficients(λ, σ, t₀, δ), 0.0, nothing, nothing, (coeffs, t) -> coeffs.δ_times_t = coeffs.δ * t)
# end

function StochasticBetaScalingEquation(λ::Float64, σ::Vector{Float64}, t₀::Float64; δ=0.0)
    coeffs = StochasticBetaScalingEquationCoefficients(λ, σ, t₀, δ)
    function update_coefficients!(coeffs::StochasticBetaScalingEquationCoefficients, t::Float64)
        @inbounds for k in eachindex(coeffs.σ)
            coeffs.δ_times_t[k] = coeffs.δ * t
        end
    end
    Nx = length(σ)
    F0 = fill(0.0, Nx)  # Initial condition for g(t)
    return StochasticBetaScalingEquation(coeffs, F0, nothing, nothing, update_coefficients!, Nx)
end

# function allocate_temporary_arrays(::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver)
#     start_time = time()
#     F_temp = Float64[]
#     F_I = Float64[]
#     temp_arrays = ModeCouplingTheory.SolverCache(F_temp, 
#                                   nothing, 
#                                   F_I, 
#                                   nothing, 
#                                   0.0, 
#                                   0.0, 
#                                   0.0, 
#                                   0.0, 
#                                   0.0, 
#                                   0.0, 
#                                   nothing, 
#                                   nothing, 
#                                   false, 
#                                   start_time
#     )
#     for _ in 1:4*solver.N
#         push!(temp_arrays.F_temp, 1.0)
#         push!(temp_arrays.F_I, 1.0)
#     end
#     return temp_arrays
# end

function allocate_temporary_arrays(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver)
    start_time = time()
    Nx = equation.Nx
    Nt = 4 * solver.N

    F_temp = Vector{Vector{Float64}}(undef, Nt)
    F_I    = Vector{Vector{Float64}}(undef, Nt)

    for t = 1:Nt
        F_temp[t] = fill(1.0, Nx)
        F_I[t]    = fill(1.0, Nx)
    end

    return ModeCouplingTheory.SolverCache(F_temp,
                                          nothing,
                                          F_I,
                                          nothing,
                                          0.0, ## type of C1
                                          0.0, ## type of C1_temp
                                          fill(1.0, Nx), ## type of C2 WILL NEED TO DO ELEM WISE DIVISION
                                          fill(1.0, Nx), ## type of C3
                                          fill(1.0, Nx),
                                          fill(1.0, Nx), ## type of F_old
                                          nothing,
                                          nothing,
                                          true,
                                          start_time)
end

# """
#     initialize_F_temp!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

# Fills the first 2N entries of the temporary arrays needed for solving the
# β-scaling equation with a an adapted Fuchs scheme.

# In particular, this initializes g(t) = (t/t₀)^-a.

# """
# function initialize_F_temp!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
#     N = solver.N
#     δt = solver.Δt / (4 * N)
#     t₀ = equation.coeffs.t₀
#     a = equation.coeffs.a

#     for it = 1:2N
#         temp_arrays.F_temp[it] = (δt * it / t₀)^-a
#     end
# end

function initialize_F_temp!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a
    Nx = equation.Nx

    for it = 1:2N
        val = (δt * it / t₀)^-a
        for k = 1:Nx
            temp_arrays.F_temp[it][k] = val
        end
    end
end

# """
#     initialize_integrals!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

# Initializes the integrals over the first 2N time points for the solution
# of the β-scaling equation, using the known critical decay law as the
# short-time asymptote.

# """
# function initialize_integrals!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
#     F_I = temp_arrays.F_I
#     N = solver.N
#     δt = solver.Δt / (4 * N)
#     t₀ = equation.coeffs.t₀
#     a = equation.coeffs.a

#     F1 = (t₀ / δt)^a / (1 - a)
#     F_I[1] = F1
#     for it = 2:2N
#         F_I[it] = F1 * (it^(1 - a) - (it - 1)^(1 - a))
#     end
# end

function initialize_integrals!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
    F_I = temp_arrays.F_I
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a
    Nx = equation.Nx

    F1 = (t₀ / δt)^a / (1 - a)

    for k = 1:Nx
        F_I[1][k] = F1
    end

    for it = 2:2N
        for k = 1:Nx
            F_I[it][k] = F1 * (it^(1 - a) - (it - 1)^(1 - a))
        end
    end
end

# """
#     update_Fuchs_parameters!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)

# Updates the parameters that are needed to solve the β-scaling equation
# numerically with Fuchs' scheme.
# """
# function update_Fuchs_parameters!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
#     N = solver.N
#     i2 = 2N
#     δt = solver.Δt / (4N)
#     F_I = temp_arrays.F_I
#     F = temp_arrays.F_temp
#     equation.update_coefficients!(equation.coeffs, δt * it)
#     λ = equation.coeffs.λ
#     σ = equation.coeffs.σ
#     δ_times_t = equation.coeffs.δ_times_t

#     temp_arrays.C1 = 2 * F_I[1]
#     temp_arrays.C2 = λ

#     c3 = -F[it-i2] * F[i2] + 2 * F[it-1] * F_I[1]
#     c3 += σ - δ_times_t
#     @inbounds for j = 2:i2
#         c3 += (F[it-j] - F[it-j+1]) * F_I[j]
#     end
#     @inbounds for j = 2:it-i2
#         c3 += (F[it-j] - F[it-j+1]) * F_I[j]
#     end
#     #@inbounds if it-i2 != i2
#     #    c3 += (F[i2] - F[i2+1]) * F_I[it-i2]
#     #end
#     temp_arrays.C3 = c3
# end

function update_Fuchs_parameters!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
    N = solver.N
    i2 = 2N
    δt = solver.Δt / (4N)
    F_I = temp_arrays.F_I
    F = temp_arrays.F_temp
    Nx = equation.Nx
    equation.update_coefficients!(equation.coeffs, δt * it)
    λ = equation.coeffs.λ
    σ = equation.coeffs.σ
    δ_times_t = equation.coeffs.δ_times_t

    a = equation.coeffs.a
    t0 = equation.coeffs.t₀

    temp_arrays.C1 = 2 * (t0 / δt)^a / (1 - a) 
    temp_arrays.C2 .= λ
    # temp_arrays.C3 .= 0.0

    for k in 1:Nx
        c3_k = -F[it - i2][k] * F[i2][k] + 2 * F[it - 1][k] * F_I[1][k]
        c3_k += σ[k] - δ_times_t[k]
        for j = 2:i2
            c3_k += (F[it - j][k] - F[it - j + 1][k]) * F_I[j][k]
        end
        for j = 2:(it - i2)
            c3_k += (F[it - j][k] - F[it - j + 1][k]) * F_I[j][k]
        end
        temp_arrays.C3[k] = c3_k
    end
end

# function update_F!(::StochasticBetaScalingEquation, ::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
#     c1 = temp_arrays.C1
#     c2 = temp_arrays.C2
#     c3 = temp_arrays.C3
#     # F_old = temp_arrays.F_temp[it]
#     temp_arrays.F_temp[it] = c1 / (2c2) - sqrt((c1 / (2c2))^2 - c3 / c2)
#     #temp_arrays.F_temp[it] = c1 \ (F_old*F_old*c2 + c3)
# end

function update_F!(::StochasticBetaScalingEquation, ::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
    C1 = temp_arrays.C1
    C2 = temp_arrays.C2
    C3 = temp_arrays.C3
    F = temp_arrays.F_temp
    Nx = length(C1)

    for k in 1:Nx
        disc = (C1[k] / (2 * C2[k]))^2 - C3[k] / C2[k]
        F[it][k] = disc ≥ 0 ? C1[k] / (2 * C2[k]) - sqrt(disc) : 0.0  # Prevent NaN
    end
end

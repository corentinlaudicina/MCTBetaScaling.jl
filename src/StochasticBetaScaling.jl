import ModeCouplingTheory.allocate_temporary_arrays
import ModeCouplingTheory.initialize_F_temp!
import ModeCouplingTheory.initialize_integrals!
import ModeCouplingTheory.update_Fuchs_parameters!
import ModeCouplingTheory.update_F!
import ModeCouplingTheory.AbstractNoKernelEquation
import ModeCouplingTheory.SolverCache
import ModeCouplingTheory.AbstractSolverCache
import ModeCouplingTheory.TimeDoublingSolver
import ModeCouplingTheory.do_time_steps!
import ModeCouplingTheory.update_integrals!

using SpecialFunctions, LinearAlgebra
include("HelperFunctions.jl")
 
struct Laplacian2D5pt
    nx::Int
    ny::Int
    dx::Float64
end

mutable struct StochasticBetaScalingEquationCoefficients{T, V}
    λ::T
    α::T
    σ::V
    t₀::T
    δ::V
    δ_times_t::V
    a::T
    b::T
    L_sys::T ## physical size of the system
end

struct StochasticBetaScalingEquation{T,A,B,C,D} <: AbstractNoKernelEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
    Nx::Int ## number of spatial sites
end

function LinearAlgebra.mul!(out::Vector{Float64}, L::Laplacian2D5pt, x::Vector{Float64})
    nx, ny, dx = L.nx, L.ny, L.dx
    dx2_inv = 1.0 / dx^2
    grid_size = (nx, ny)

    # Reshape vectors into 2D arrays using the grid size
    X = reshape(x, grid_size)
    O = reshape(out, grid_size)

    @inbounds for I in CartesianIndices(grid_size)
        i, j = Tuple(I)

        # Wrap-around (periodic boundary conditions)
        ip = i == nx ? 1 : i + 1
        im = i == 1  ? nx : i - 1
        jp = j == ny ? 1 : j + 1
        jm = j == 1  ? ny : j - 1

        # Sum neighbors and subtract center

        O[I] = (
            X[im, j] + X[ip, j] +
            X[i, jm] + X[i, jp] -
            4 * X[I]
        ) * dx2_inv
    end
    return out
end

function get_center_term(L::Laplacian2D5pt, x::Vector{Float64})
    nx, ny = L.nx, L.ny
    grid_size = (nx, ny)
    dx = L.dx
    # Reshape vectors into 2D arrays using the grid size
    X = reshape(x, grid_size)
    
    out = similar(x)
    O = reshape(out, grid_size)

    @inbounds for I in CartesianIndices(grid_size)
        i, j = Tuple(I)
        O[I] = - 4 * X[i, j] / dx^2
    end
    return out
end

function get_non_center_terms(L::Laplacian2D5pt, x::Vector{Float64})
    nx, ny = L.nx, L.ny
    grid_size = (nx, ny)
    dx = L.dx
    # Reshape vectors into 2D arrays using the grid size
    X = reshape(x, grid_size)
    out = similar(x)
    O = reshape(out, grid_size)

    @inbounds for I in CartesianIndices(grid_size)
        i, j = Tuple(I)

        # Wrap-around (periodic boundary conditions)
        ip = i == nx ? 1 : i + 1
        im = i == 1  ? nx : i - 1
        jp = j == ny ? 1 : j + 1
        jm = j == 1  ? ny : j - 1

        O[I] = (
            X[im, j] + X[ip, j] +
            X[i, jm] + X[i, jp]
        ) / dx^2
    end
    return out
end


function Base.show(io::IO, ::MIME"text/plain", p::StochasticBetaScalingEquation)
    println(io, "MCT beta-scaling object:")
    println(io, "   σ - δ t + λ (g(t))² = ∂ₜ∫g(t-τ)g(τ)dτ")
    println(io, "with real-valued parameters.")
end


function StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    exponent_func = (x) -> (SpecialFunctions.gamma(1 - x) / SpecialFunctions.gamma(1 - 2x)) * SpecialFunctions.gamma(1 - x) - λ
    a = regula_falsi(0.2, 0.3, exponent_func)
    b = -regula_falsi(-0.5, -0.1, exponent_func)
    return StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, δ*0.0, a, b, L_sys)
end

function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64; δ=zeros(length(σ)))
    coeffs = StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    function update_coefficients!(coeffs::StochasticBetaScalingEquationCoefficients, t::Float64)
        @inbounds for k in eachindex(coeffs.σ)
            coeffs.δ_times_t[k] = coeffs.δ[k] * t
        end
    end
    Nx = length(σ)
    F0 = fill(0.0, Nx)  # Initial condition for g(t)
    return StochasticBetaScalingEquation(coeffs, F0, nothing, nothing, update_coefficients!, Nx)
end

function allocate_temporary_arrays(eq::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver)
    start_time = time()
    F_temp = Vector{Float64}[]
    F_I = Vector{Float64}[]
    temp_arrays = ModeCouplingTheory.SolverCache(
                                  F_temp, 
                                  nothing, 
                                  F_I, 
                                  nothing, 
                                  zeros(eq.Nx), # c1
                                  zeros(eq.Nx), # c1_temp
                                  zeros(eq.Nx), # c2
                                  zeros(eq.Nx), # c3
                                  zeros(eq.Nx), # temp_vec
                                  zeros(eq.Nx), # F_old
                                  nothing, #temp_mat
                                  nothing, # linsolvecache
                                  true, 
                                  start_time
    )
    Nx = eq.Nx
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, fill(0.0, Nx))
        push!(temp_arrays.F_I, fill(0.0, Nx))
    end
    return temp_arrays
end


"""
    initialize_F_temp!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

Fills the first 2N entries of the temporary arrays needed for solving the
β-scaling equation with a an adapted Fuchs scheme.

In particular, this initializes g(t) = (t/t₀)^-a.

"""
function initialize_F_temp!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a
    # Need to fix this with correction for alpha!!!!!!!!!
    for it = 1:2N
        temp_arrays.F_temp[it] .= (δt * it / t₀)^-a
    end
end


"""
    initialize_integrals!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

Initializes the integrals over the first 2N time points for the solution
of the β-scaling equation, using the known critical decay law as the
short-time asymptote.

"""
function initialize_integrals!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)
    F_I = temp_arrays.F_I
    N = solver.N
    δt = solver.Δt / (4 * N)
    t₀ = equation.coeffs.t₀
    a = equation.coeffs.a
    # Need to fix this with correction for alpha!!!!!!!!!

    F1 = (t₀ / δt)^a / (1 - a)
    F_I[1] .= F1
    for it = 2:2N
        val = F1 * (it^(1 - a) - (it - 1)^(1 - a))
        F_I[it] .= val
    end
end

"""
    update_Fuchs_parameters!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)

Updates the parameters that are needed to solve the β-scaling equation
numerically with Fuchs' scheme.
"""
function update_Fuchs_parameters!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
    N = solver.N
    i2 = it ÷ 2
    δt = solver.Δt / (4N)
    F_I = temp_arrays.F_I
    F = temp_arrays.F_temp
    equation.update_coefficients!(equation.coeffs, δt * it)
    λ = equation.coeffs.λ
    α = equation.coeffs.α
    σ = equation.coeffs.σ
    δ_times_t = equation.coeffs.δ_times_t

    temp_arrays.C1 = 2 * F_I[1]
    temp_arrays.C2 .= λ

    c3 = -F[it-i2] .* F[i2] .+ 2 .* F[it-1] .* F_I[1]
    c3 .+= σ .- δ_times_t

    ## compute Laplacian of F[it]
    # dx = equation.coeffs.L_sys / sqrt(equation.Nx)
    # mylaplacian = Laplacian2D5pt(sqrt(equation.Nx), sqrt(equation.Nx), dx)
    # LinearAlgebra.mul!(temp_arrays.temp_vec, mylaplacian, F[it])

    @inbounds for j = 2:i2
        c3 .+= (F[it-j] .- F[it-j+1]) .* F_I[j]
    end
    @inbounds for j = 2:it-i2
        c3 .+= (F[it-j] .- F[it-j+1]) .* F_I[j]
    end
    @inbounds if it-i2 != i2
       c3 .+= (F[i2] .- F[i2+1]) .* F_I[it-i2]
    end
    temp_arrays.C3 .= c3
end


function update_F!(equation::StochasticBetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)
    α = equation.coeffs.α
    tolerance = solver.tolerance
    max_iterations = solver.max_iterations

    Lap = Laplacian2D5pt(sqrt(equation.Nx), sqrt(equation.Nx), equation.coeffs.L_sys / sqrt(equation.Nx))

    c1 = temp_arrays.C1
    c2 = temp_arrays.C2
    c3 = temp_arrays.C3
    # F_old = temp_arrays.F_temp[it]
    for i in eachindex(equation.coeffs.σ)
        disc = (c1[i] / (2 * c2[i]))^2 - c3[i] / c2[i]
        temp_arrays.F_temp[it][i] = c1[i] / (2 * c2[i]) - sqrt(disc) 
    end

    # if (α == 0)
    #     return
    # end
    # extrapolation guess
    for i in eachindex(equation.coeffs.σ)
       temp_arrays.F_temp[it][i] = 2temp_arrays.F_temp[it-1][i] - temp_arrays.F_temp[it-2][i] 
    end
    newF = temp_arrays.temp_vec
    for i in eachindex(equation.coeffs.σ)
        newF[i] = temp_arrays.F_temp[it][i]
    end
    iterations = 0
    passed = false
    λ = equation.coeffs.λ

    while !passed 
        non_center_terms = α * get_non_center_terms(Lap, temp_arrays.F_temp[it])
        center_terms = -α * get_center_term(Lap, temp_arrays.F_temp[it])
        for i in eachindex(equation.coeffs.σ)
            # _b = -c1[i]/2 - center_terms[i]/2
            # λ = c2[i]
            # _a = c3[i] + non_center_terms[i]
            # temp_func = x -> 2*_b*x + λ*x^2 + _a
            # newF[i] = regula_falsi(10.0, 20.0, temp_func)

            # # Coupling
            disc = ((c1[i] + center_terms[i]) / (2 * c2[i]))^2 - (c3[i] + non_center_terms[i]) / c2[i]
            newF[i] =(c1[i] + center_terms[i]) / (2 * c2[i]) - sqrt(disc)

            # if abs(newF[i] - temp_arrays.F_temp[it][i]) < tolerance*abs(temp_arrays.F_temp[it][i]) 
            #     passed = true
            # end
        end

        # check if all i satisfy reltol
        max_rel_err = 10000000.0
        for i in eachindex(equation.coeffs.σ)
            rel_err = abs(newF[i] - temp_arrays.F_temp[it][i]) / abs(temp_arrays.F_temp[it][i])
            max_rel_err = min(max_rel_err, rel_err)
        end
        if max_rel_err < tolerance
            passed = true
        end


        for i in eachindex(equation.coeffs.σ)
            temp_arrays.F_temp[it][i] = newF[i]
        end

        if passed 
            break 
        end

        if iterations > max_iterations
            error("Maximum iterations reached without convergence for time step $it.")
        end
        iterations += 1
    end
    #temp_arrays.F_temp[it] = c1 \ (F_old*F_old*c2 + c3)
end



"""
    do_time_steps!(equation::MemoryEquation, solver::TimeDoublingSolver, kernel::MemoryKernel, temp_arrays::SolverCache)

Solves the equation on the time points with index 2N+1 until 4N, for each point doing a recursive iteration
to find the solution to the nonlinear equation C1 F  = -C2 M(F) + C3.
"""
function do_time_steps!(equation::Union{BetaScalingEquation, StochasticBetaScalingEquation}, solver::TimeDoublingSolver, kernel, temp_arrays::AbstractSolverCache)
    N = solver.N
    F_temp = temp_arrays.F_temp
    tolerance = solver.tolerance
    for it = (2N+1):(4N)
        error = typemax(Float64)
        F_old = temp_arrays.F_old
        update_Fuchs_parameters!(equation, solver, temp_arrays, it)
        update_F!(equation, solver, temp_arrays, it)
        # @show error
        if !temp_arrays.inplace
            F_old = F_temp[it]
        else
            F_old .= F_temp[it]
        end
        # @show iterations, error
        update_integrals!(temp_arrays, equation, it)
    end
    return
end


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
using FFTW, Statistics

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
    L_sys::T
    A1::T           
end

struct StochasticBetaScalingEquation{T,A,B,C,D} <: AbstractNoKernelEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
    Nx::Int ## number of spatial sites
end

function StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    exponent_func = (x) -> (SpecialFunctions.gamma(1 - x) / SpecialFunctions.gamma(1 - 2x)) * SpecialFunctions.gamma(1 - x) - λ
    a = regula_falsi(0.2, 0.3, exponent_func)
    b = -regula_falsi(-0.5, -0.1, exponent_func)
    A1 = 1/(2 *(a * π / sin(a*π)-λ))
    return StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, δ*0.0, a, b, L_sys, A1)
end

function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64; δ=zeros(length(σ)))
    coeffs = StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    function update_coefficients!(coeffs::StochasticBetaScalingEquationCoefficients, t::Float64)
        @inbounds for k in eachindex(coeffs.σ)
            coeffs.δ_times_t[k] = coeffs.δ[k] * t
        end
    end
    Nx = length(σ)
    F0 = fill(0.0, Nx)
    return StochasticBetaScalingEquation(coeffs, F0, nothing, nothing, update_coefficients!, Nx)
end






function Base.show(io::IO, ::MIME"text/plain", p::StochasticBetaScalingEquation)
    println(io, "MCT Stochastic beta-scaling object:")
    println(io, "   σ - δ t + λ (g(t))² = ∂ₜ∫g(t-τ)g(τ)dτ")
    println(io, "with real-valued parameters.")
end




function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64; δ=zeros(length(σ)), A1::Float64=1.0)
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



function allocate_temporary_arrays(eq::StochasticBetaScalingEquation,
                                   solver::ModeCouplingTheory.TimeDoublingSolver)
    start_time = time()
    F_temp = Vector{Float64}[]
    F_I    = Vector{Float64}[]
    Nx     = eq.Nx
    nside  = Int(sqrt(Nx))
    @assert nside*nside == Nx "Nx must be a perfect square (2D grid)."

    # Build spectral cache once
    Lx = eq.coeffs.L_sys
    Ly = eq.coeffs.L_sys

    temp_arrays = ModeCouplingTheory.SolverCache(
        F_temp,
        nothing,
        F_I,
        nothing,
        zeros(Nx),   # c1
        zeros(Nx),   # c1_temp
        zeros(Nx),   # c2
        zeros(Nx),   # c3
        zeros(Nx),   # temp_vec
        zeros(Nx),   # F_old
        nothing,     # temp_mat
        nothing,   # solve_cache
        true,
        start_time
    )
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, fill(0.0, Nx))
        push!(temp_arrays.F_I,    fill(0.0, Nx))
    end
    return temp_arrays
end



"""
    initialize_F_temp!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

Fills the first 2N entries of the temporary arrays needed for solving the
β-scaling equation with a an adapted Fuchs scheme.

In particular, this initializes g(t) = (t/t₀)^-a.

"""
function initialize_F_temp!(equation::StochasticBetaScalingEquation,
                            solver::ModeCouplingTheory.TimeDoublingSolver,
                            temp_arrays::ModeCouplingTheory.SolverCache)
    N  = solver.N
    δt = solver.Δt / (4N)
    t₀ = equation.coeffs.t₀
    a  = equation.coeffs.a
    A1 = equation.coeffs.A1
    σ  = equation.coeffs.σ

    # g(x, t_i) = (t_i/t0)^(-a) + A1 * σ(x) * (t_i/t0)^(a)
    for it = 1:2N
        τ = (δt * it) / t₀
        base = τ^(-a)
        corr = (τ^a) .* (A1 .* σ)
        @. temp_arrays.F_temp[it] = base + corr
    end
end


"""
    initialize_integrals!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

Initializes the integrals over the first 2N time points for the solution
of the β-scaling equation, using the known critical decay law as the
short-time asymptote.

"""
function initialize_integrals!(equation::StochasticBetaScalingEquation,
                               solver::ModeCouplingTheory.TimeDoublingSolver,
                               temp_arrays::SolverCache)
    F_I = temp_arrays.F_I
    N   = solver.N
    δt  = solver.Δt / (4N)
    t₀  = equation.coeffs.t₀
    a   = equation.coeffs.a
    A1  = equation.coeffs.A1
    σ   = equation.coeffs.σ

    # Precompute scalar factors
    fac_minus = (δt / t₀)^(-a) / (1 - a)
    fac_plus  = (δt / t₀)^( a) / (1 + a)

    # k = 1..2N bins; each F_I[k] is a length-Nx vector (site-dependent due to σ)
    for k = 1:2N
        inc_minus = fac_minus * (k^(1 - a) - (k-1)^(1 - a))
        inc_plus  = fac_plus  * (k^(1 + a) - (k-1)^(1 + a))
        @. F_I[k] = inc_minus + A1 * σ * inc_plus
    end
end



"""
    update_Fuchs_parameters!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache, it::Int)

Updates the parameters that are needed to solve the β-scaling equation
numerically with Fuchs' scheme.
"""
function update_Fuchs_parameters!(equation::StochasticBetaScalingEquation,
                                  solver::ModeCouplingTheory.TimeDoublingSolver,
                                  temp_arrays::ModeCouplingTheory.SolverCache,
                                  it::Int)
    N   = solver.N
    i2  = it ÷ 2
    δt  = solver.Δt / (4N)

    F_I = temp_arrays.F_I         # vector moments dG_k (sitewise due to σ-correction)
    F   = temp_arrays.F_temp      # g at stored times

    # advance δ * t
    equation.update_coefficients!(equation.coeffs, δt * it)

    λ         = equation.coeffs.λ
    σ         = equation.coeffs.σ
    δ_times_t = equation.coeffs.δ_times_t

    # C1 = 2 dG1  (sitewise)
    temp_arrays.C1 .= 2 .* F_I[1]
    # (C2 not used by the Picard solve, but keep for compatibility)
    temp_arrays.C2 .= λ
    c3 = temp_arrays.C3
    # C̃_i = Σ_i + g_{i-i2} g_{i2} - 2 g_{i-1} dG1 + (δ t_i - σ)
    c3 .= F[it - i2] .* F[i2] .- 2 .* F[it - 1] .* F_I[1]
    c3 .+= (δ_times_t .- σ)

    # Σ_i = 2 * sum_{j=2}^{i2} (g_{i-j+1}-g_{i-j}) dG_j  + middle term if i odd
    s = temp_arrays.temp_vec
    s .= zero(eltype(s))
    @inbounds for j = 2:i2
        s .+= (F[it - j + 1] .- F[it - j]) .* F_I[j]
    end
    c3 .+= 2 .* s
    if isodd(it)
        jmid = it - i2               # == i2 + 1
        c3 .+= (F[i2 + 1] .- F[i2]) .* F_I[jmid]
    end
end


# Periodic 4-neighbour sum: out = sum of N,S,E,W of x
function neighbor_sum4!(out::Vector{Float64}, x::Vector{Float64}, nside::Int)
    X = reshape(x, nside, nside)
    O = reshape(out, nside, nside)
    @inbounds for j in 1:nside, i in 1:nside
        ip = (i == nside) ? 1 : i+1
        im = (i == 1)     ? nside : i-1
        jp = (j == nside) ? 1 : j+1
        jm = (j == 1)     ? nside : j-1
        O[i,j] = X[im,j] + X[ip,j] + X[i,jm] + X[i,jp]
    end
    return out
end



function update_F!(equation::StochasticBetaScalingEquation,
                   solver::ModeCouplingTheory.TimeDoublingSolver,
                   temp::ModeCouplingTheory.SolverCache,
                   it::Int)

    λ  = equation.coeffs.λ
    α  = equation.coeffs.α
    C1 = temp.C1          # length Nx
    C3 = temp.C3

    # grid/spacing
    Nx    = length(C1)
    nside = Int(round(sqrt(Nx)))
    @assert nside*nside == Nx "Nx must be a perfect square grid."
    dx    = equation.coeffs.L_sys / nside
    αeff  = α / (dx*dx)
    z     = 2.0           # 2D: 4 neighbours → center shift = 2*αeff

    # time slices
    g  = temp.F_temp[it]
    g1 = temp.F_temp[it-1]
    g2 = temp.F_temp[it-2]

    # extrapolated start
    @inbounds @simd for i in 1:Nx
        g[i] = 2*g1[i] - g2[i]
    end

    # g0 bracket (reuse cache F_old)
    g0 = temp.F_old
    @inbounds @simd for i in 1:Nx
        g0[i] = root_no_nabla(C1[i], C3[i], λ)
    end

    # b_i = C1/2 + z*αeff  (reuse c1_temp)
    b = temp.C1_temp
    @inbounds @simd for i in 1:Nx
        b[i] = 0.5*C1[i] + z*αeff
    end

    # work buffer (sumNN, then overwritten with a_i)
    work = temp.temp_vec

    maxit   = (solver.max_iterations > 0) ? solver.max_iterations : 200
    tol_rel = (solver.tolerance > 0) ? solver.tolerance : 1e-9

    iters = 0
    while true
        # sumNN = Σ_4nn g
        neighbor_sum4!(work, g, nside)

        # work := a_i = C3 - αeff * sumNN
        @inbounds for i in 1:Nx
            work[i] = C3[i] - αeff*work[i]
        end

        # sitewise regula falsi, in-place update of g
        maxrel = 0.0
        @inbounds for i in 1:Nx
            gi_old = g[i]
            ok, gi_try = quad_root_safe(λ, b[i], work[i], gi_old)
            gi_new = ok ? gi_try : regula_falsi_illinois(λ, b[i], work[i], gi_old, g0[i];
                                                    tol=1e-12, maxit=200)


            # gi_old = g[i]
            # gi_new = regula_falsi_quad(λ, b[i], work[i], gi_old, g0[i];
            #                            tol=1e-12, maxit=200)
            g[i] = gi_new
            denom = abs(gi_old); if denom < 1e-14; denom = 1e-14; end
            rel = abs(gi_new - gi_old) / denom
            if rel > maxrel; maxrel = rel; end
        end

        iters += 1
        if (maxrel <= tol_rel) 
            return
        end
        if (iters >= maxit)
            error("Max iterations reached without convergence")
            return
        end
    end
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


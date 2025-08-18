



 
struct Laplacian2D5pt
    nx::Int
    ny::Int
    dx::Float64
end


mutable struct StochasticBetaScalingEquationCoefficients{T, V, DIMS}
    λ::T
    α::T
    σ::V
    t₀::T
    δ::V
    δ_times_t::V
    a::T
    b::T
    L_sys::NTuple{DIMS, T}  # System size in 2D
    A1::T           
end

struct StochasticBetaScalingEquation{T,A,B,C,D, DIMS} <: AbstractNoKernelEquation
    coeffs::T
    F₀::A
    K₀::B
    kernel::C
    update_coefficients!::D
    ns::NTuple{DIMS, Int}  # Number of sites in each dimension
end

function StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    exponent_func = (x) -> (SpecialFunctions.gamma(1 - x) / SpecialFunctions.gamma(1 - 2x)) * SpecialFunctions.gamma(1 - x) - λ
    a = regula_falsi(0.2, 0.3, exponent_func)
    b = -regula_falsi(-0.5, -0.1, exponent_func)
    A1 = 1/(2 *(a * π / sin(a*π)-λ))
    return StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, δ*0.0, a, b, L_sys, A1)
end


"""
    StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::NTuple{DIMS, Float64}, ns::NTuple{DIMS, Int}; δ=zeros(length(σ))) where DIMS

Creates a stochastic β-scaling equation object for a system with specified parameters.
This equation is defined as:

    σ(x) - δ t + λ (g(x,t))² + α ∇²g(x,t) = ∂ₜ∫g(x, t-τ)g(x, τ)dτ

where `σ` is a vector of site-dependent noise, `δ` is a vector of damping coefficients,
`λ` is the coupling constant, `α` is the diffusion coefficient, `t₀` is a reference time, and `L_sys` is the system size in each dimension.
"""
function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::NTuple{DIMS, Float64}, ns::NTuple{DIMS, Int}; δ=zeros(length(σ))) where DIMS
    @assert allequal(L_sys) "L must be equal in all dimensions."
    @assert allequal(ns) "ns must be a tuple of equal dimensions."
    @assert length(σ) == prod(ns) "Length of σ must match the product of ns dimensions."
    coeffs = StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    function update_coefficients!(coeffs::StochasticBetaScalingEquationCoefficients, t::Float64)
        @inbounds for k in eachindex(coeffs.σ)
            coeffs.δ_times_t[k] = coeffs.δ[k] * t
        end
    end
    @assert prod(ns) == length(σ)
    Nx = length(σ)
    F0 = fill(0.0, Nx)
    return StochasticBetaScalingEquation(coeffs, F0, nothing, nothing, update_coefficients!, ns)
end

"""
    StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64, ns::Int; δ=zeros(length(σ)))

Creates a stochastic β-scaling equation object for a system with specified parameters.
This is a convenience constructor 1D systems.
"""
function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64, ns::Int; δ=zeros(length(σ)))
    return StochasticBetaScalingEquation(λ, α, σ, t₀, (L_sys,), (ns, ); δ=δ)
end



function Base.show(io::IO, ::MIME"text/plain", p::StochasticBetaScalingEquation)
    println(io, "MCT Stochastic beta-scaling object:")
    println(io, "   σ(x) - δ t + λ (g(x,t))² + α * ∇²g(x,t) = ∂ₜ∫g(x, t-τ)g(x, τ)dτ")
    println(io, "   λ = $(p.coeffs.λ), α = $(p.coeffs.α), t₀ = $(p.coeffs.t₀), δ = $(p.coeffs.δ)")
    println(io, "   L_sys = $(p.coeffs.L_sys), with $(p.ns) sites in each dimension.")
end



function allocate_temporary_arrays(eq::StochasticBetaScalingEquation,
                                   solver::ModeCouplingTheory.TimeDoublingSolver)
    start_time = time()
    F_temp = Vector{Float64}[]
    F_I    = Vector{Float64}[]
    Ntot     = prod(eq.ns)

    temp_arrays = ModeCouplingTheory.SolverCache(
        F_temp,
        nothing,
        F_I,
        nothing,
        zeros(Ntot),   # c1
        zeros(Ntot),   # c1_temp
        zeros(Ntot),   # c2
        zeros(Ntot),   # c3
        zeros(Ntot),   # temp_vec
        zeros(Ntot),   # F_old
        nothing,     # temp_mat
        nothing,   # solve_cache
        true,
        start_time
    )
    for _ in 1:4*solver.N
        push!(temp_arrays.F_temp, fill(0.0, Ntot))
        push!(temp_arrays.F_I,    fill(0.0, Ntot))
    end
    return temp_arrays
end



"""
    initialize_F_temp!(equation::BetaScalingEquation, solver::ModeCouplingTheory.TimeDoublingSolver, temp_arrays::ModeCouplingTheory.SolverCache)

Fills the first 2N entries of the temporary arrays needed for solving the
β-scaling equation with a an adapted Fuchs scheme.

In particular, this initializes g(t) = (t/t₀)^-a + A1 * σ(x) * (t/t0)^(a)

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

# periodic 1-dim neighbor sum: out = sum of N,S of x
function neighbor_sum!(out::Vector{Float64}, x::Vector{Float64}, ns::NTuple{1, Int})
    X = reshape(x, ns...)
    O = reshape(out, ns...)
    @inbounds for i in 1:ns[1]
        ip = (i == nside) ? 1 : i+1
        im = (i == 1)     ? nside : i-1
        O[i,j] = X[im] + X[ip]
    end
    return out
end

# Periodic 2*dims-neighbour sum: out = sum of N,S,E,W of x
function neighbor_sum!(out::Vector{Float64}, x::Vector{Float64}, ns::NTuple{2, Int})
    X = reshape(x, ns...)
    O = reshape(out, ns...)
    nsidey, nsidex = ns
    @inbounds for j in 1:nsidey, i in 1:nsidex
        ip = (i == nsidex) ? 1 : i+1
        im = (i == 1)     ? nsidex : i-1
        jp = (j == nsidey) ? 1 : j+1
        jm = (j == 1)     ? nsidey : j-1
        O[i,j] = X[im,j] + X[ip,j] + X[i,jm] + X[i,jp]
    end
    return out
end

# Periodic 3*dims-neighbour sum: out = sum of N,S,E,W,U,D of x
function neighbor_sum!(out::Vector{Float64}, x::Vector{Float64}, ns::NTuple{3, Int})
    X = reshape(x, ns...)
    O = reshape(out, ns...)
    nsidez, nsidey, nsidex = ns
    @inbounds for k in 1:nsidez, j in 1:nsidey, i in 1:nsidex
        ip = (i == nsidex) ? 1 : i+1
        im = (i == 1)     ? nsidex : i-1
        jp = (j == nsidey) ? 1 : j+1
        jm = (j == 1)     ? nsidey : j-1
        kp = (k == nsidez) ? 1 : k+1
        km = (k == 1)     ? nsidez : k-1
        O[i,j,k] = X[im,j,k] + X[ip,j,k] + X[i,jm,k] + X[i,jp,k] +
                   X[i,j,km] + X[i,j,kp]
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
    Ntot    = length(C1)
    ns = equation.ns
    L_sys = equation.coeffs.L_sys
    dx    = L_sys[1] / ns[1]
 
    αeff  = α / (dx*dx)
    z     = length(ns)           # number of neighbors / 2

    # time slices
    g  = temp.F_temp[it]
    g1 = temp.F_temp[it-1]
    g2 = temp.F_temp[it-2]

    # extrapolated start
    @inbounds @simd for i in 1:Ntot
        g[i] = 2*g1[i] - g2[i]
    end

    # g0 bracket (reuse cache F_old)
    g0 = temp.F_old
    @inbounds @simd for i in 1:Ntot
        g0[i] = root_no_nabla(C1[i], C3[i], λ)
    end

    # b_i = C1/2 + z*αeff  (reuse c1_temp)
    b = temp.C1_temp
    @inbounds @simd for i in 1:Ntot
        b[i] = 0.5*C1[i] + z*αeff
    end

    # work buffer (sumNN, then overwritten with a_i)
    work = temp.temp_vec

    maxit   = (solver.max_iterations > 0) ? solver.max_iterations : 200
    tol_rel = (solver.tolerance > 0) ? solver.tolerance : 1e-9

    iters = 0
    while true
        # sumNN = Σ_4nn g
        neighbor_sum!(work, g, ns)

        # work := a_i = C3 - αeff * sumNN
        @inbounds for i in 1:Ntot
            work[i] = C3[i] - αeff*work[i]
        end

        # analytic sol. otherwise sitewise Ill. regula falsi, in-place update of g
        maxrel = 0.0
        @inbounds for i in 1:Ntot
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


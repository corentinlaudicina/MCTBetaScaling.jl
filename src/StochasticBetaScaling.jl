"""
    NeighborTable

Precomputed periodic nearest-neighbour index table for a `D`-dimensional hypercubic lattice holding `N` sites in total. Column `i` of `idx` lists the linear indices of the `2D` neighbours of site `i`, ordered `(+1, -1)` along dimension 1, then dimension 2, etc.

The table is built once per equation (see [`build_neighbor_table`](@ref)) and reused by `neighbor_sum!` on every root-find iteration of every timestep, so the per-call cost of the discrete Laplacian is a pure gather with no boundary branches and no dimension-specific
code path.
"""
struct NeighborTable
    D::Int
    N::Int
    idx::Matrix{Int32}   # size (2D, N); idx[:,i] = linear indices of site i's 2D neighbours
end

"""
    build_neighbor_table(ns::NTuple{D,Int}) where D

Builds the [`NeighborTable`](@ref) of a periodic lattice with `ns[d]` sites along dimension `d`, for any number of dimensions `D`.
"""
function build_neighbor_table(ns::NTuple{D,Int}) where D
    N    = prod(ns)
    cart = CartesianIndices(ns)
    lin  = LinearIndices(ns)
    idx  = Matrix{Int32}(undef, 2D, N)
    @inbounds for i in 1:N
        ci = Tuple(cart[i])
        for d in 1:D
            n  = ns[d]
            cp = ntuple(k -> k==d ? mod1(ci[k]+1, n) : ci[k], Val(D))
            cm = ntuple(k -> k==d ? mod1(ci[k]-1, n) : ci[k], Val(D))
            idx[2d-1, i] = lin[CartesianIndex(cp)]
            idx[2d,   i] = lin[CartesianIndex(cm)]
        end
    end
    return NeighborTable(D, N, idx)
end

# shortcut: `update_F!` solves each time step by Jacobi sweeps, whose count grows ~linearly
# with αeff = α/dx². Measured on a 64×64 lattice (λ = 0.75, σ uniform in ±0.17, t_max = 1e10,
# max_iterations = 10⁴): αeff = 16 / 40 / 80 take 79 / 160 / 342 s, αeff = 160 stops with
# "Max iterations". Physical couplings (α ≈ Γ ~ 0.1 with dx ≳ 1) sit far below; above this
# the constructor warns. Conjugate gradients on the linearized step would lift the limit.
const ALPHA_EFF_WARN = 50.0


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
    nbrs::NeighborTable    # periodic neighbour indices, derived from ns once at construction
end

function StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    a, b = _exponents(λ)
    A1 =1/(2 *(a * π / sin(a*π)-λ))
    return StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, δ*0.0, a, b, L_sys, A1)
end


"""
    StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::NTuple{DIMS, Float64}, ns::NTuple{DIMS, Int}; δ=zeros(length(σ))) where DIMS

Creates a stochastic β-scaling equation object for a system with specified parameters. This equation is defined as:

    σ(x) - δ t + λ (g(x,t))² + α ∇²g(x,t) = ∂ₜ∫g(x, t-τ)g(x, τ)dτ

where `σ` is a vector of site-dependent noise, `δ` is a vector of damping coefficients, `λ` is the coupling constant, `α` is the diffusion coefficient, `t₀` is a reference time, and `L_sys` is the system size in each dimension.
"""
function StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::NTuple{DIMS, Float64}, ns::NTuple{DIMS, Int}; δ=zeros(length(σ))) where DIMS
    @assert allequal(L_sys) "L must be equal in all dimensions."
    @assert allequal(ns) "ns must be a tuple of equal dimensions."
    @assert length(σ) == prod(ns) "Length of σ must match the product of ns dimensions."
    αeff = α / (L_sys[1] / ns[1])^2
    αeff > ALPHA_EFF_WARN && @warn "Strong lattice coupling αeff = α/dx² = $αeff > $ALPHA_EFF_WARN: the " *
        "per-step Jacobi iteration slows down ~linearly in αeff and stops converging somewhere between " *
        "αeff ≈ 80 and 160. Couplings this strong are probably unphysical (α ≈ Γ ~ 0.1, dx ≳ 1)."
    coeffs = StochasticBetaScalingEquationCoefficients(λ, α, σ, t₀, δ, L_sys)
    function update_coefficients!(coeffs::StochasticBetaScalingEquationCoefficients, t::Float64)
        @inbounds for k in eachindex(coeffs.σ)
            coeffs.δ_times_t[k] = coeffs.δ[k] * t
        end
    end
    Nx = length(σ)
    F0 = fill(0.0, Nx)
    nbrs = build_neighbor_table(ns)
    return StochasticBetaScalingEquation(coeffs, F0, nothing, nothing, update_coefficients!, ns, nbrs)
end

"""
    StochasticBetaScalingEquation(λ::Float64, α::Float64, σ::Vector{Float64}, t₀::Float64, L_sys::Float64, ns::Int; δ=zeros(length(σ)))

Creates a stochastic β-scaling equation object for a system with specified parameters. This is a convenience constructor 1D systems.
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

# Both per-site loops below (the neighbour gather, and the root find in `update_F!`) are
# embarrassingly parallel over lattice sites, but spawning the tasks costs on the order of
# 10 µs -- more than the whole loop takes on a small lattice, and paid once per root-find
# iteration. So each loop is split into one chunk per thread only above the site count
# below; under it, it runs as a single chunk on the calling thread. Site i is computed the
# same way either way, so results are bitwise independent of the thread count.
#
# The threshold was measured end-to-end by A/B-ing it (`solve`, 4 threads, Apple aarch64,
# Julia 1.12): threading costs 1.34x at 16k sites and 1.08x at 33k, breaks even near 45k,
# then pays 1.25x at 58k and 1.40x at 66k. Both loops spawn once per iteration, so they
# share one threshold -- gating them separately only opens a window where one pays the
# spawn cost without the other earning it back. The crossover depends on the machine (core
# count, cache, memory bandwidth), hence a `Ref` rather than a constant: set
# `MCTBetaScaling.THREAD_MIN_SITES[] = n` to retune it, or to 0 to always thread.
const THREAD_MIN_SITES = Ref(40_000)

@inline function _site_chunks(N::Int, min_sites::Int = THREAD_MIN_SITES[])
    (Threads.nthreads() == 1 || N < min_sites) && return 1
    return min(Threads.nthreads(), N)
end
@inline _chunk_lo(c::Int, N::Int, nchunks::Int) = div((c-1)*N, nchunks) + 1
@inline _chunk_hi(c::Int, N::Int, nchunks::Int) = div(c*N,     nchunks)

# out[i] = Σ over the 2D nearest neighbours of site i, for sites lo:hi.
@inline function _neighbor_sum_chunk!(out::Vector{Float64}, x::Vector{Float64},
                                     idx::Matrix{Int32}, twoD::Int, lo::Int, hi::Int)
    @inbounds for i in lo:hi
        s = 0.0
        @simd for k in 1:twoD
            s += x[idx[k,i]]
        end
        out[i] = s
    end
    return nothing
end

# Periodic 2D-neighbour sum for any number of dimensions: the neighbour-sum term of the
# discrete periodic Laplacian. The neighbour indices are precomputed once per equation
# (see `build_neighbor_table`), so this is a pure gather: 2D loads per site, no boundary
# branches, and one code path for every D. The D = 1, 2, 3 hand-written versions this
# replaces are kept as oracles in test/test_laplacian.jl.
function neighbor_sum!(out::Vector{Float64}, x::Vector{Float64}, nbrs::NeighborTable)
    idx, twoD, N = nbrs.idx, 2*nbrs.D, nbrs.N
    nchunks = _site_chunks(N)
    if nchunks == 1
        _neighbor_sum_chunk!(out, x, idx, twoD, 1, N)
    else
        Threads.@threads for c in 1:nchunks
            _neighbor_sum_chunk!(out, x, idx, twoD,
                                 _chunk_lo(c, N, nchunks), _chunk_hi(c, N, nchunks))
        end
    end
    return out
end

# One chunk of the sitewise root find of λ g² - 2 b g - a = 0 (a = `work`), updating g in
# place and returning the largest relative change over the chunk. Site i reads only b[i]
# and work[i] and writes only g[i], so chunks are independent. Without a real root, g
# takes the vertex b/λ, the point where the quadratic comes closest to zero.
@inline function _root_find_chunk!(g::Vector{Float64}, b::Vector{Float64},
                                  work::Vector{Float64}, λ::Float64, lo::Int, hi::Int)
    maxrel = 0.0
    @inbounds for i in lo:hi
        gi_old = g[i]
        ok, gi_try = quad_root_safe(λ, b[i], work[i], gi_old)
        gi_new = ok ? gi_try : b[i] / λ
        g[i] = gi_new
        denom = abs(gi_old); if denom < 1e-14; denom = 1e-14; end
        rel = abs(gi_new - gi_old) / denom
        if rel > maxrel; maxrel = rel; end
    end
    return maxrel
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

    # b_i = C1/2 + z*αeff  (reuse c1_temp)
    b = temp.C1_temp
    @inbounds @simd for i in 1:Ntot
        b[i] = 0.5*C1[i] + z*αeff
    end

    # work buffer (sumNN, then overwritten with a_i)
    work = temp.temp_vec

    maxit   = (solver.max_iterations > 0) ? solver.max_iterations : 200
    tol_rel = (solver.tolerance > 0) ? solver.tolerance : 1e-9

    # per-chunk maxima for the site loop below, allocated once per call
    nchunks   = _site_chunks(Ntot)
    chunk_max = Vector{Float64}(undef, nchunks)

    iters = 0
    while true
        # sumNN = Σ_2Dnn g
        neighbor_sum!(work, g, equation.nbrs)

        # work := a_i = C3 - αeff * sumNN
        @inbounds for i in 1:Ntot
            work[i] = C3[i] - αeff*work[i]
        end

        # sitewise analytic root, in-place update of g.
        # The per-chunk maxima are reduced after the loop, so nothing is written to a
        # shared scalar from several threads.
        if nchunks == 1
            chunk_max[1] = _root_find_chunk!(g, b, work, λ, 1, Ntot)
        else
            Threads.@threads for c in 1:nchunks
                chunk_max[c] = _root_find_chunk!(g, b, work, λ,
                                                 _chunk_lo(c, Ntot, nchunks),
                                                 _chunk_hi(c, Ntot, nchunks))
            end
        end
        maxrel = maximum(chunk_max)

        iters += 1
        if (maxrel <= tol_rel) 
            return
        end
        if (iters >= maxit)
            error("Max iterations ($maxit) reached without convergence; lattice coupling " *
                  "αeff = α/dx² = $αeff (see ALPHA_EFF_WARN)")
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
    for it = (2N+1):(4N)
        update_Fuchs_parameters!(equation, solver, temp_arrays, it)
        update_F!(equation, solver, temp_arrays, it)
        # F_old is not read anywhere for these equations:
        # F_old = temp_arrays.F_old
        # if !temp_arrays.inplace
        #     F_old = temp_arrays.F_temp[it]
        # else
        #     F_old .= temp_arrays.F_temp[it]
        # end
        update_integrals!(temp_arrays, equation, it)
    end
    return
end


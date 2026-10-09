# Tests for the dimension-generic periodic lattice Laplacian (neighbour-sum term).
#
# The library used to carry one hand-written `neighbor_sum!` per dimension (D = 1, 2, 3);
# those are now replaced by a single table-driven method. The D = 2 and D = 3 versions are
# reproduced verbatim below as known-correct oracles so the generalization stays pinned to
# the old numerics. (The old D = 1 version never ran -- it referenced an undefined `nside`
# and indexed a 1D array as `O[i,j]` -- so D = 1 is checked against a hand-written oracle
# instead.)

using Test, Random
using MCTBetaScaling: NeighborTable, build_neighbor_table, neighbor_sum!,
                      _neighbor_sum_chunk!, _root_find_chunk!, _site_chunks,
                      THREAD_MIN_SITES

# ---------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------

# Copied verbatim from the pre-generalization library code (only the name changed).
function neighbor_sum_old!(out::Vector{Float64}, x::Vector{Float64}, ns::NTuple{2, Int})
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

# Copied verbatim from the pre-generalization library code (only the name changed).
function neighbor_sum_old!(out::Vector{Float64}, x::Vector{Float64}, ns::NTuple{3, Int})
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

# Dimension-agnostic, obviously-correct (but slow) reference: walk every Cartesian shift.
# Comparisons against it use `≈`: `neighbor_sum!` marks its 2D-term reduction `@simd`, which
# lets the compiler sum the neighbours pairwise rather than left-to-right, so the two differ
# at the last bit or two. `==` is used below only where it is exact regardless of order
# (repeated calls, dyadic values) -- that is where it actually tests for a data race.
function neighbor_sum_reference(x::Vector{Float64}, ns::NTuple{D,Int}) where D
    X   = reshape(x, ns...)
    out = similar(X)
    for ci in CartesianIndices(ns)
        c = Tuple(ci)
        s = 0.0
        for d in 1:D
            up   = ntuple(k -> k == d ? mod1(c[k] + 1, ns[d]) : c[k], D)
            down = ntuple(k -> k == d ? mod1(c[k] - 1, ns[d]) : c[k], D)
            s += X[CartesianIndex(up)]
            s += X[CartesianIndex(down)]
        end
        out[ci] = s
    end
    return vec(out)
end

generic_sum(x, ns) = neighbor_sum!(similar(x), x, build_neighbor_table(ns))

@info "test_laplacian running with $(Threads.nthreads()) thread(s)"

# ---------------------------------------------------------------------------------------
# 1. Regression against the old D = 2 and D = 3 implementations
# ---------------------------------------------------------------------------------------
@testset "regression vs. old D=$(length(ns)) implementation, ns=$ns" for ns in
        [(8, 8), (17, 17), (5, 5, 5), (9, 9, 9)]
    Random.seed!(42 + length(ns))
    x   = randn(prod(ns))
    old = neighbor_sum_old!(similar(x), x, ns)
    new = generic_sum(x, ns)
    @test new ≈ old
    @test new ≈ neighbor_sum_reference(x, ns)
end

# ---------------------------------------------------------------------------------------
# 2. D = 1: hand-written oracle (the old method was broken and could not be used)
# ---------------------------------------------------------------------------------------
@testset "D=1 periodic chain" begin
    n = 6
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    expected = [x[mod1(i - 1, n)] + x[mod1(i + 1, n)] for i in 1:n]
    @test expected == [2.0 + 32.0, 1.0 + 4.0, 2.0 + 8.0, 4.0 + 16.0, 8.0 + 32.0, 16.0 + 1.0]
    @test generic_sum(x, (n,)) == expected

    Random.seed!(7)
    y = randn(n)
    @test generic_sum(y, (n,)) ≈ [y[mod1(i - 1, n)] + y[mod1(i + 1, n)] for i in 1:n]
end

# ---------------------------------------------------------------------------------------
# 3. D = 1..8: invariants (no old implementation exists above D = 3)
# ---------------------------------------------------------------------------------------
@testset "invariants at D=$D" for D in 1:8
    n  = max(3, floor(Int, 4096^(1 / D)))       # keep N_tot ≲ 4096, and n ≥ 3
    ns = ntuple(_ -> n, D)
    N  = prod(ns)
    tbl = build_neighbor_table(ns)

    @test tbl.D == D
    @test tbl.N == N
    @test size(tbl.idx) == (2D, N)
    @test all(1 .<= tbl.idx .<= N)
    # +1 and -1 neighbours along a dimension are distinct for n ≥ 3, so each site has
    # exactly 2D distinct neighbours, none of which is the site itself
    @test all(i -> length(unique(tbl.idx[:, i])) == 2D, 1:N)
    @test all(i -> !(Int32(i) in tbl.idx[:, i]), 1:N)
    # the adjacency relation is symmetric
    @test all(i -> all(j -> Int32(i) in tbl.idx[:, j], tbl.idx[:, i]), 1:N)

    # constant field c -> 2D*c everywhere
    c = 0.375
    @test generic_sum(fill(c, N), ns) == fill(2D * c, N)

    # single nonzero site -> exactly 2D sites receive exactly 1.0, all others 0
    for site in (1, N ÷ 3 + 1, N)
        x = zeros(N); x[site] = 1.0
        out = generic_sum(x, ns)
        @test count(!iszero, out) == 2D
        @test all(out[j] == 1.0 for j in tbl.idx[:, site])
        @test sum(out) == 2D
    end

    # matches the brute-force Cartesian-shift reference on a random field
    Random.seed!(100 + D)
    x = randn(N)
    @test generic_sum(x, ns) ≈ neighbor_sum_reference(x, ns)

    # the operator is self-adjoint: ⟨y, Ax⟩ = ⟨x, Ay⟩
    y = randn(N)
    @test sum(y .* generic_sum(x, ns)) ≈ sum(x .* generic_sum(y, ns))
end

# ---------------------------------------------------------------------------------------
# 4. Non-cubic lattices (the table supports them even though the equation asserts cubic)
# ---------------------------------------------------------------------------------------
@testset "non-cubic lattice ns=$ns" for ns in [(3, 5), (4, 6, 3), (3, 4, 5, 6)]
    Random.seed!(sum(ns))
    x = randn(prod(ns))
    @test generic_sum(x, ns) ≈ neighbor_sum_reference(x, ns)
end

# ---------------------------------------------------------------------------------------
# 5. Threading: results must not depend on the thread count or on scheduling
# ---------------------------------------------------------------------------------------
@testset "threaded neighbor_sum! is deterministic" begin
    ns  = (32, 32, 32)
    tbl = build_neighbor_table(ns)
    N   = prod(ns)
    # force the parallel path regardless of the machine's measured break-even, so that this
    # is a test of the threaded code and not of the threshold
    old_min = THREAD_MIN_SITES[]
    THREAD_MIN_SITES[] = 0
    @test _site_chunks(N) == (Threads.nthreads() > 1 ? Threads.nthreads() : 1)
    Random.seed!(2024)
    x    = randn(N)
    ref  = neighbor_sum_reference(x, ns)
    out1 = neighbor_sum!(similar(x), x, tbl)
    @test out1 ≈ ref
    # bitwise-identical across repeated calls: any race in the threaded loop would show up
    # here as a differing site, independent of summation order
    for _ in 1:5
        @test neighbor_sum!(similar(x), x, tbl) == out1
    end
    # ... and bitwise-identical to running the same kernel as a single serial chunk, which
    # is what the sub-threshold path does
    serial = similar(x)
    _neighbor_sum_chunk!(serial, x, tbl.idx, 2 * tbl.D, 1, N)
    @test serial == out1
    # writing into a buffer that already holds garbage must not leak it
    out2 = fill(NaN, N)
    @test neighbor_sum!(out2, x, tbl) == out1

    # and the sub-threshold path agrees with the threaded one bitwise
    THREAD_MIN_SITES[] = typemax(Int)
    @test _site_chunks(N) == 1
    @test neighbor_sum!(similar(x), x, tbl) == out1
    THREAD_MIN_SITES[] = old_min
end

@testset "root find is invariant under chunk decomposition" begin
    # The root-find loop in update_F! is split into one chunk per thread and the per-chunk
    # maxima reduced afterwards; splitting must change neither g nor the reduced maximum.
    N = 5000
    Random.seed!(99)
    λ    = 0.75
    g    =  1.0 .+ 0.1 .* randn(N)
    b    =  2.0 .+ 0.1 .* randn(N)
    work = -1.0 .+ 0.1 .* randn(N)

    g_whole = copy(g)
    max_whole = _root_find_chunk!(g_whole, b, work, λ, 1, N)

    for nchunks in (2, 3, 4, 7, 64)
        g_split = copy(g)
        maxima  = [_root_find_chunk!(g_split, b, work, λ,
                                     div((c - 1) * N, nchunks) + 1, div(c * N, nchunks))
                   for c in 1:nchunks]
        @test g_split == g_whole
        @test maximum(maxima) == max_whole
    end

    # no real root (b² + λa < 0): g takes the vertex b/λ
    g1 = [0.5]
    _root_find_chunk!(g1, [1.0], [-2.0], λ, 1, 1)
    @test g1[1] == 1.0 / λ
end

# ---------------------------------------------------------------------------------------
# 6. Wiring into StochasticBetaScalingEquation + end-to-end solve at several D
# ---------------------------------------------------------------------------------------
@testset "equation carries the neighbour table (D=$D)" for D in 1:4
    n  = 4
    ns = ntuple(_ -> n, D)
    Ls = ntuple(_ -> 10.0, D)
    Random.seed!(11)
    σ  = 0.01 .* randn(prod(ns))
    eq = StochasticBetaScalingEquation(0.75, 0.1, σ, 0.001, Ls, ns)
    @test eq.nbrs isa NeighborTable
    @test eq.nbrs.D == D
    @test eq.nbrs.N == prod(ns)
    @test eq.nbrs.idx == build_neighbor_table(ns).idx
end

@testset "solve runs at D=$D (D=1 used to throw UndefVarError)" for D in 1:3
    ns = ntuple(_ -> 4, D)
    Ls = ntuple(_ -> 10.0, D)
    Random.seed!(3 + D)
    σ  = 0.01 .* randn(prod(ns))
    eq = StochasticBetaScalingEquation(0.75, 0.1, σ, 0.001, Ls, ns)
    sol = solve(eq, TimeDoublingSolver(t_max = 1e2, Δt = 1e-6, N = 8, tolerance = 1e-10))
    F = reduce(hcat, sol.F)
    @test all(isfinite, F)
    @test size(F, 1) == prod(ns)
end

# ---------------------------------------------------------------------------------------
# 7. Threading, cross-process: a multi-threaded solve must equal a single-threaded one.
#    Only meaningful when this process itself has more than one thread.
# ---------------------------------------------------------------------------------------
# THREAD_MIN_SITES is forced to 0 so that the multi-threaded run takes the parallel path
# for both the gather and the root find even on this small, quick-to-solve lattice.
const SOLVE_SNIPPET = """
using MCTBetaScaling, ModeCouplingTheory, Random
MCTBetaScaling.THREAD_MIN_SITES[] = 0
ns = (32, 32); Ls = (10.0, 10.0)
Random.seed!(31337)
σ  = 0.01 .* randn(prod(ns))
eq = StochasticBetaScalingEquation(0.75, 0.1, σ, 0.001, Ls, ns)
sol = solve(eq, TimeDoublingSolver(t_max = 1e-3, Δt = 1e-6, N = 8, tolerance = 1e-10))
write(ARGS[1], reinterpret(UInt8, reduce(vcat, sol.F)))
"""

@testset "solve is identical single- vs multi-threaded" begin
    if Threads.nthreads() == 1
        @info "skipping cross-thread-count solve check: only 1 thread available"
        @test Threads.nthreads() == 1   # placeholder so the testset is never empty
    else
        mine   = tempname()
        serial = tempname()
        script = tempname() * ".jl"
        write(script, SOLVE_SNIPPET)
        proj = Base.active_project()
        run(`$(Base.julia_cmd()) -t $(Threads.nthreads()) --project=$proj $script $mine`)
        run(`$(Base.julia_cmd()) -t 1                     --project=$proj $script $serial`)
        a = reinterpret(Float64, read(mine))
        b = reinterpret(Float64, read(serial))
        @test length(a) > 0
        @test a == b
        foreach(rm, (mine, serial, script))
    end
end

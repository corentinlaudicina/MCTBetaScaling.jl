


module MCTBetaScaling
    using ModeCouplingTheory, SpecialFunctions, LinearAlgebra

    export BetaScalingEquation
    export StochasticBetaScalingEquation
    export compute_lambda
    export compute_lambda_multicomponent
    export compute_gamma, gamma_matrices, glass_sweep, sweep_summary
    export BipolarMCTKernel, memory_functional, steady_state_map
    export stability_matrix, perron_pair, steady_state, find_critical_packing
    export c_PY, S_PY, dcf_PY, dcf_derivatives, dcf_spline
    export PYHyperspheres, solve_PY, dcf_PY_numeric
    export compute_C
    export compute_mu
    export compute_DeltaSigma2
    export S0_PY, extrapolate_S0

    for file in ["HelperFunctions.jl", "BetaScaling.jl", "StochasticBetaScaling.jl", "lambda.jl",
                 "Structure.jl", "BipolarKernel.jl", "Stability.jl", "SteadyState.jl",
                 "Gamma.jl", "SeparationParameter.jl", "DeltaSigma2.jl"]
        include(file)
    end

end # module

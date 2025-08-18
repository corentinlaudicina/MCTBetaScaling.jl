

module MCTBetaScaling
    using ModeCouplingTheory, SpecialFunctions, LinearAlgebra
    
    export BetaScalingEquation
    export StochasticBetaScalingEquation
    export compute_lambda

    for file in ["HelperFunctions.jl", "BetaScaling.jl", "StochasticBetaScaling.jl", "lambda.jl"]
        include(file)
    end
   
end # module

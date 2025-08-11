
"""
Package to solve mode-coupling theory like equations
"""
module MCTBetaScaling
    using ModeCouplingTheory, SpecialFunctions, LinearAlgebra
    export BetaScalingEquation
    export StochasticBetaScalingEquation

    for file in ["HelperFunctions.jl", "BetaScaling.jl", "StochasticBetaScaling.jl"]
        include(file)
    end
   
end # module

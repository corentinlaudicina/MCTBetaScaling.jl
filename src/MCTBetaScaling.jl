
"""
Package to solve mode-coupling theory like equations
"""
module MCTBetaScaling
    using ModeCouplingTheory, SpecialFunctions
    using LinearAlgebra
    export BetaScalingEquation

    for file in ["HelperFunctions.jl", "BetaScaling.jl", "lambda.jl"]
        include(file)
    end
    


end # module

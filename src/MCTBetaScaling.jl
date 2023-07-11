
"""
Package to solve mode-coupling theory like equations
"""
module MCTBetaScaling
    using ModeCouplingTheory, SpecialFunctions
    export BetaScalingEquation

    for file in ["HelperFunctions.jl", "BetaScaling.jl"]
        include(file)
    end
    


end # module

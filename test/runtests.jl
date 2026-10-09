using Test, ModeCouplingTheory, MCTBetaScaling


for target in ["beta", "lambda", "lambda_multicomponent", "gamma", "separation_parameter", "DeltaSigma2", "laplacian", "stochastic"]
    @testset "$target" begin
        include("test_$target.jl")
    end
end

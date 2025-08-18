using Test, ModeCouplingTheory, MCTBetaScaling


for target in ["beta", "lambda"]
    @testset "$target" begin
        include("test_$target.jl")
    end
end

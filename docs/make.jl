using Documenter
using MCTBetaScaling

makedocs(
    sitename = "MCTBetaScaling",
    format = Documenter.HTML(),
    modules = [MCTBetaScaling]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/IlianPihlajamaa/MCTBetaScaling.jl.git",
    devbranch = "main"
)
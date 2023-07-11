using Documenter
using MCTBetaScaling


push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "MCTBetaScaling",
    pages = [
        "Introduction" => "index.md",
        "API Reference" => "API.md",
     ],
     format = Documenter.HTML(prettyurls = false)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/IlianPihlajamaa/MCTBetaScaling.jl.git",
    devbranch = "main"
)
using GaussChebyshevFracture
using Documenter

DocMeta.setdocmeta!(GaussChebyshevFracture, :DocTestSetup, :(using GaussChebyshevFracture); recursive=true)

makedocs(;
    modules=[GaussChebyshevFracture],
    authors="Antoine Jacquey <antoine.jacquey@polymtl.ca> and contributors",
    repo="https://github.com/ajacquey/GaussChebyshevFracture.jl/blob/{commit}{path}#{line}",
    sitename="GaussChebyshevFracture.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ajacquey.github.io/GaussChebyshevFracture.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ajacquey/GaussChebyshevFracture.jl",
    devbranch="main",
)

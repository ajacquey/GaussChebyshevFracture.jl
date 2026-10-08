```@meta
CurrentModule = GaussChebyshevFracture
```

# [GaussChebyshevFracture](@id home-section)

```@eval
using Markdown
using Pkg

VERSION = Pkg.dependencies()[Base.UUID("3ef21536-876f-437b-8e16-8b3c2d5a66c3")].version

github_badge = "[![Github](https://img.shields.io/badge/ajacquey-GaussChebyshevFracture.jl-blue.svg?logo=github)](https://github.com/ajacquey/GaussChebyshevFracture.jl)"

version_badge = "![v$VERSION](https://img.shields.io/badge/version-v$(replace("$VERSION", "-" => "--"))-green.svg)"

if get(ENV, "DOCUMENTER_BUILD_PDF", "") == ""
    Markdown.parse("$github_badge $version_badge")
else
    Markdown.parse("""
    -----

    On Github: [GaussChebyshevFracture.jl](https://github.com/ajacquey/GaussChebyshevFracture.jl)

    Version: $VERSION

    -----

    """)
end
```

*A set of numerical methods (Gauss-Chebyshev quadrature) for coupled fracture problems*

Documentation for [GaussChebyshevFracture](https://github.com/ajacquey/GaussChebyshevFracture.jl).

## Overview

!!! tip "Useful reference"
    The notation and algorithms implemented were drawn from the following reference:
    - Viesca, R.C. and Garagash, D. *Numerical methods for coupled fracture problems*. Journal of the Mechanics and Physics of Solids. 113, 13-34, 2018. [doi:10.1016/j.jmps.2018.01.008](https://www.sciencedirect.com/science/article/abs/pii/S0022509617306853).

This package provides utilities to solve crack problems which are represented as integral equations of the (non-dimensional) form:

```math
\sigma\left(x\right) = \sigma_{0}\left(x\right) + \frac{1}{\pi} \int_{-1}^{+1} K\left(x - s\right) \frac{d\delta}{ds}\, ds.
```

where $x$ is the coorditate along which a straight crack lies, $\sigma$ is the stress component whose traction is oriented eigher normal or parallel to the crack face, $\delta$ is the relative displacement (difference in displacement of each side of the crack face, opening or slip) and $\sigma_{0}$ is the background stress or the stress in the absence of relative displacement. $K\left(\xi\right)$ is a kernel whose most singular term will be $\frac{1}{\xi}$. 

This formulation arise in problems of in-plane, anti-plane, or axisymmetric rupture (opening or sliding) of a linear elastic material. For a non-exhaustive list of problems, we refer to the tests.

## Installation

You can install `GaussChebyshevFracture` directly from the `Pkg` REPL:

```
pkg> add GaussChebyshevFracture
```

## Illustrative example

For this example, let's look at a crack subject to uniform stress drop, as described in section 3.1 of [Viesca and Garagash (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0022509617306853).

Let's consider a traction-free crack of length $2a$ under a uniform far-field loading $\sigma_{0}$.
The dimensional form of the problem is

```math
\sigma\left(x\right) = \sigma_{0} - \frac{\mu^{\prime}}{2 \pi} \int_{-a}^{+a} \frac{1}{x - s} \frac{d \delta}{d s} \, ds,
```

where $\sigma\left(x\right) = 0$ for $\left|x\right| < a$ and the modulus $\mu^{\prime} = \frac{\mu}{1 - \nu}$ for modes I and II (plane strain), and $\mu^{\prime} = \mu$ for mode III, $\mu$ being the shear modulus and $\nu$ the Poisson's ratio of the medium hosting the crack. $\delta$ is the displacement discontinuity on the crack surface.

After non-dimensionalization with $\frac{x}{a} \rightarrow x$ and $\frac{\delta \mu^{\prime}}{2 \sigma_{0}a} \rightarrow \delta$, the problem becomes:

```math
1 = \frac{1}{\pi} \int_{-1}^{+1} \frac{1}{x - s}\frac{d\delta}{ds}\, ds, \text{ for } \left|x\right| < 1.
```
Anticipating, due to the crack singularity, that $\frac{d\delta}{ds} \sim \frac{1}{\sqrt{1 - \left|s\right|}}$ as $\left|s\right| \rightarrow 1$, we look for a solution of the form $\frac{d\delta}{ds} = \frac{F\left(s\right)}{\sqrt{1 - s^{2}}}$, the Gaussian quadrature for the singular integral implies the discretization:

```math
1 \approx \frac{1}{\pi} \sum_{j=1}^{n} w_{j} \frac{F\left(s_{j}\right)}{x_{i} - s_{j}}.
```
For this configuration, we can use 

```@example uniform_stress_drop
using GaussChebyshevFracture
using Plots
using LinearAlgebra

# Number of quadrature points
n = 100

# Gauss-Chebyshec quadrature of 1st kind
gc = GaussChebyshev(n, 1)
```

The structure `gc` is the core of the package and provides several utilities, including the quadrature weights `w`, the evaluation points `x` and the quadrature points `s`.

The discretization yields $n$ unknowns $F\left(s_{j}\right)$ and only $n-1$ equations.
This under-determinancy is resolved by an additional constraint. 
Classically, such a constraint is the requirement of zero cumulative relative displacement, which can be expressed as:

```math
0 = \delta\left(1\right) - \delta\left(-1\right) = \int_{-1}^{+1} \frac{d\delta}{ds}\, ds = \int_{-1}^{+1} \frac{F\left(s\right)}{\sqrt{1 - s^{2}}}\, ds \approx \sum_{j=1}^{n} w_{j} F\left(s_{j}\right).
```

We use this additional constraint to solve for the slip.

```@example uniform_stress_drop
# We solve the following system of equations:
# A F = b
# Right-end side b
b = ones(n-1)
b = vcat(b, 0.0)

# Matrix A
A = [gc.w[j] / (π * (gc.x[i] - gc.s[j])) for i in eachindex(gc.x), j in eachindex(gc.s)]
A = vcat(A, gc.w')

# Solutions for F(s)
F = A \ b
```

The known solution of this problem is given by:

```math
\delta\left(x\right) = \sqrt{1 - x^{2}}.
```

```@example uniform_stress_drop
# Solution slip
δ = GaussChebyshevFracture.u(gc, F)
# Analytical solution slip
sol_δ = sqrt.(1 .- gc.x.^2)

norm((δ.-sol_δ)./sol_δ)
```

```@example uniform_stress_drop
# Plot solutions
plot(gc.x, sol_δ; xlabel="x", ylabel="δ", label="Analytical")
scatter!(gc.x, δ,markersize=2; label="Numerical")

```

## Index

```@index
```

```@autodocs
Modules = [GaussChebyshevFracture]
```

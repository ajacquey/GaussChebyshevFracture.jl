module TestUniformStressDrop

using GaussChebyshevFracture
using Statistics
using Test

@testset "Uniform stress drop" begin
    # Number of quadrature points
    n = 100

    # Gauss-Chebyshec quadrature
    gc = GaussChebyshev(n, 1)

    # Right-end side
    b = ones(n-1)
    b = vcat(b, 0.0)

    # matrix
    A = [gc.w[j] / (π * (gc.x[i] - gc.s[j])) for i in eachindex(gc.x), j in eachindex(gc.s)]
    A = vcat(A, gc.w')

    # Solutions for F(s)
    F = A \ b

    # Solution slip
    δ = GaussChebyshevFracture.u(gc, F)
    sol_δ = sqrt.(1 .- gc.x.^2)

    # Solution slip gradient
    ∇δ = GaussChebyshevFracture.∇u(gc, F)
    sol_∇δ = -gc.x ./ sqrt.(1 .- gc.x.^2)

    # Check if numerical solutions is the same as analytical solutions
    @test isapprox(δ, sol_δ) && isapprox(∇δ, sol_∇δ)
end
end

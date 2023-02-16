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
  δ = gc.S * F
  sol = sqrt.(1 .- gc.x.^2)

  # Error
  err = mean(abs.((sol .- δ) ./ sol))
  # Error less than 1%
  @test err < 0.01
end
end

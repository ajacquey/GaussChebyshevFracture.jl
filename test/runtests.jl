using GaussChebyshevFracture
using Test

@testset "GaussChebyshevFracture.jl" begin
    include("uniform_stress_drop.jl")
    include("dugdale_barenblatt.jl")
end

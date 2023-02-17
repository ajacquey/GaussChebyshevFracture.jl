using GaussChebyshevFracture
using Test

@testset "GaussChebyshevFracture.jl" begin
    include("uniform_stress_drop.jl")
    include("dugdale_barenblatt.jl")
    include("injection_constant_friction.jl")
end

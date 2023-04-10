module TestDugdaleBarenBlatt

using GaussChebyshevFracture
using SpecialFunctions
using NLsolve
using Statistics
using Test

@testset "Injection constant friction" begin
    @testset "T = 0.2" begin
        # Fault stress parameter
        T = 0.2

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Matrix
        A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- erfc.(abs.(x[n+1] * gc.x)) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * abs.(gc.x) .* exp.(-x[n+1]^2 * gc.x.^2) / sqrt(π)
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.9125140; rtol=1.0e-02)
    end
    @testset "T = 0.4" begin
        # Fault stress parameter
        T = 0.4

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Matrix
        A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- erfc.(abs.(x[n+1] * gc.x)) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * abs.(gc.x) .* exp.(-x[n+1]^2 * gc.x.^2) / sqrt(π)
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.0252441; rtol=1.0e-02)
    end
    @testset "T = 0.6" begin
        # Fault stress parameter
        T = 0.6

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Matrix
        A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- erfc.(abs.(x[n+1] * gc.x)) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * abs.(gc.x) .* exp.(-x[n+1]^2 * gc.x.^2) / sqrt(π)
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.60122936; rtol=1.0e-02)
    end
    @testset "T = 0.8" begin
        # Fault stress parameter
        T = 0.8

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Matrix
        A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- erfc.(abs.(x[n+1] * gc.x)) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * abs.(gc.x) .* exp.(-x[n+1]^2 * gc.x.^2) / sqrt(π)
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.28337740; rtol=1.0e-02)
    end
end
end
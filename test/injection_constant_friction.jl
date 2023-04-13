module TestConstantFriction

using GaussChebyshevFracture
using SpecialFunctions
using NLsolve
using Statistics
using Test

@testset "Injection constant friction" begin
    @testset "2D, T = 0.2" begin
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
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.9125140; rtol=1.0e-02)
    end
    @testset "2D, T = 0.4" begin
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
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.0252441; rtol=1.0e-02)
    end
    @testset "2D, T = 0.6" begin
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
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.60122936; rtol=1.0e-02)
    end
    @testset "2D, T = 0.8" begin
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
        F = res.zero[1:n]
        λ = res.zero[end]

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.28337740; rtol=1.0e-02)
    end
    @testset "3D, T = 0.001" begin
        # Fault stress parameter
        T = 0.001

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ abs.(x[n+1])
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.0 / sqrt(2 * T); rtol=2.0e-02)
    end
    @testset "3D, T = 0.01" begin
        # Fault stress parameter
        T = 0.01

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ abs.(x[n+1])
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.0 / sqrt(2 * T); rtol=2.0e-02)
    end
    @testset "3D, T = 0.1" begin
        # Fault stress parameter
        T = 0.1

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ abs.(x[n+1])
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 1.0 / sqrt(2 * T); rtol=4.0e-02)
    end
    @testset "3D, T = 2.0" begin
        # Fault stress parameter
        T = 2.0

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ x[n+1]
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.1), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.5* exp((2.0 - Base.MathConstants.γ - T) / 2); rtol=5.0e-02)
    end
    @testset "3D, T = 4.0" begin
        # Fault stress parameter
        T = 4.0

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ x[n+1]
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.1), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.5* exp((2.0 - Base.MathConstants.γ - T) / 2); rtol=1.0e-02)
    end
    @testset "3D, T = 8.0" begin
        # Fault stress parameter
        T = 8.0

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        function k(u)
            return 2.0 * sqrt(u) / (1.0 + u)
        end

        # Matrix
        A = [gc.w[j] / π * (ellipe(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2)/ (gc.s[j] - gc.x[i]) + ellipk(k((1.0 + gc.x[i]) / (1.0 + gc.s[j])).^2) / (2.0 + gc.s[j] + gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= T .- expint.(1, x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 2 * exp.(-x[n+1]^2 * (1.0 .+ gc.x).^2 / 4) ./ x[n+1]
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.01), method = :newton)

        # Solutions
        F = res.zero[1:n]
        λ = res.zero[end]
        δ = GaussChebyshevFracture.u(gc, F)

        # Check if numerical solutions is correct (check λ)
        @test converged(res) && isapprox(λ, 0.5* exp((2.0 - Base.MathConstants.γ - T) / 2); rtol=1.0e-02)
    end
end
end
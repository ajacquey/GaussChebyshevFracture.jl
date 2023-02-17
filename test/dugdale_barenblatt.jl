module TestDugdaleBarenBlatt

using GaussChebyshevFracture
using NLsolve
using Statistics
using Test

@testset "Dugdale-Barenblatt" begin
    @testset "c = 0.25" begin
        # Critical value position
        c = 0.25

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Index at critical value
        idx = findall(gc.x .> c)[1]

        # Forcing vector
        f = zeros(n + 1)
        f[abs.(gc.x) .> c] .= 1.0

        # Matrix
        A = [gc.w[j] / (π * (gc.x[i] - gc.s[j])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= x[n+1] .- f .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 1.0
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        δ = GaussChebyshevFracture.u(gc, F)
        sol_r = 1 - 2 * asin(c) / π
        sol_δ = c / π * log.(abs.((c^2 .- gc.x.^2) ./ (sqrt(1 - c^2) .- sqrt.(1 .- gc.x.^2)).^2)) .+ gc.x / π .* log.(abs.((gc.x .* sqrt(1 - c^2) .- c * sqrt.(1 .- gc.x.^2)) ./ (gc.x .* sqrt(1 - c^2) .+ c * sqrt.(1 .- gc.x.^2)))) 

        # Check if numerical solutions is the same as analytical solutions
        @test converged(res) && isapprox(r, sol_r; rtol=1.0e-02) && isapprox(δ, sol_δ; rtol=2.0e-02)
    end
    @testset "c = 0.5" begin
        # Critical value position
        c = 0.5

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Index at critical value
        idx = findall(gc.x .> c)[1]

        # Forcing vector
        f = zeros(n + 1)
        f[abs.(gc.x) .> c] .= 1.0

        # Matrix
        A = [gc.w[j] / (π * (gc.x[i] - gc.s[j])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= x[n+1] .- f .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 1.0
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        δ = GaussChebyshevFracture.u(gc, F)
        sol_r = 1 - 2 * asin(c) / π
        sol_δ = c / π * log.(abs.((c^2 .- gc.x.^2) ./ (sqrt(1 - c^2) .- sqrt.(1 .- gc.x.^2)).^2)) .+ gc.x / π .* log.(abs.((gc.x .* sqrt(1 - c^2) .- c * sqrt.(1 .- gc.x.^2)) ./ (gc.x .* sqrt(1 - c^2) .+ c * sqrt.(1 .- gc.x.^2)))) 

        # Check if numerical solutions is the same as analytical solutions
        @test converged(res) && isapprox(r, sol_r; rtol=1.0e-02) && isapprox(δ, sol_δ; rtol=1.0e-02)
    end
    @testset "c = 0.75" begin
        # Critical value position
        c = 0.75

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Index at critical value
        idx = findall(gc.x .> c)[1]

        # Forcing vector
        f = zeros(n + 1)
        f[abs.(gc.x) .> c] .= 1.0

        # Matrix
        A = [gc.w[j] / (π * (gc.x[i] - gc.s[j])) for i in eachindex(gc.x), j in eachindex(gc.s)]

        # Residual function
        function res!(R, x)
            R[1:n+1] .= x[n+1] .- f .- (A * x[1:n])
        end

        # Jacobian function
        function jac!(J, x)
            J[1:n+1,1:n] .= -A
            J[:,n+1] .= 1.0
        end

        # Newton solve
        res = nlsolve(res!, jac!, vcat(zeros(n), 0.5), method = :newton)

        # Solutions
        r = res.zero[end]
        F = res.zero[1:n]
        δ = GaussChebyshevFracture.u(gc, F)
        sol_r = 1 - 2 * asin(c) / π
        sol_δ = c / π * log.(abs.((c^2 .- gc.x.^2) ./ (sqrt(1 - c^2) .- sqrt.(1 .- gc.x.^2)).^2)) .+ gc.x / π .* log.(abs.((gc.x .* sqrt(1 - c^2) .- c * sqrt.(1 .- gc.x.^2)) ./ (gc.x .* sqrt(1 - c^2) .+ c * sqrt.(1 .- gc.x.^2)))) 

        # Check if numerical solutions is the same as analytical solutions
        @test converged(res) && isapprox(r, sol_r; rtol=1.0e-02) && isapprox(δ, sol_δ; rtol=1.0e-02)
    end
end
end
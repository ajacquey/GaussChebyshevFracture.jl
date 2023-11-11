module TestSlipWeakeningFriction

using GaussChebyshevFracture
using SpecialFunctions
using NLsolve
using Statistics
using Test
using UnicodePlots
using DelimitedFiles

@testset "Injection slip-weakening friction" begin
    @testset "fᵣ/fₚ = 0.6, Δp/σ₀ = 0.5, τ₀/τₚ = 0.55" begin
        # Friction ration
        fᵣ_fₚ = 0.6

        # Overpressure
        Δp_σ₀ = 0.5

        # Initial shear stress
        τ₀_τₚ = 0.55

        # Number of quadrature points
        n = 200

        # Gauss-Chebyshec quadrature
        gc = GaussChebyshev(n, 2)

        # Matrix
        A = [gc.w[j] / (π * (gc.s[j] - gc.x[i])) for i in eachindex(gc.x), j in eachindex(gc.s)]
        
        # Linear slip-weakening friction
        function friction(F::Vector{Float64})::Vector{Float64}
            δ = GaussChebyshevFracture.u(gc, F)
            f_fₚ = similar(δ)
            fill!(f_fₚ, fᵣ_fₚ)
            f_fₚ[δ .<= 1.0 - fᵣ_fₚ] .= 1 .- δ[δ .<= 1.0 - fᵣ_fₚ]
            return f_fₚ
        end
        function friction_derivative(F::Vector{Float64})::Matrix{Float64}
            δ = GaussChebyshevFracture.u(gc, F)
            df_fₚ = zeros(size(gc.S))
            df_fₚ[δ .<= 1.0 - fᵣ_fₚ, :] .= -gc.S[δ .<= 1.0 - fᵣ_fₚ, :]
            return df_fₚ
        end

        function fluid_pressure(x::Vector{Float64}, a::Float64, t::Float64)
            if t == 0.0
                return 0.0
            else
                erfc.(abs.(x * a / sqrt(t)))
            end
        end
        function dfluid_pressure(x::Vector{Float64}, a::Float64, t::Float64)
            if t == 0.0
                return 0.0
            else
                return -2 * abs.(gc.x / sqrt(t)) .* exp.(-(x * a).^2 / t) / sqrt(π)
            end
        end

        time = collect(range(0.0, stop=15.0, length=200).^2)
        a = zeros(length(time))
        δ₀ = zeros(length(time))
        F = zeros(n)
        aᵢ = 0.0

        for i in eachindex(time)
            println("Time: ", time[i])
            

            # Residual function
            function res!(R, x)
                f_fₚ = friction(x[1:n])
                # R[1:n+1] .= (A * x[1:n]) .- x[n+1] * Δp_σ₀ .* ((f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* erfc.(abs.(x[n+1] * gc.x / time[i])))
                R[1:n+1] .= abs(x[n+1]) * Δp_σ₀ * ((f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* fluid_pressure(gc.x, x[n+1], time[i])) .- (A * x[1:n])
            end

            # Jacobian function
            function jac!(J, x)
                f_fₚ = friction(x[1:n])
                df_fₚ = friction_derivative(x[1:n])
                # J[1:n+1,1:n] .= A .- x[n+1] * Δp_σ₀ .* df_fₚ .* (1 / Δp_σ₀ .- erfc.(abs.(x[n+1] * gc.x / time[i])))
                # J[:,n+1] .= -Δp_σ₀* ((f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* erfc.(abs.(x[n+1] * gc.x / time[i])) .+ 2 * x[n+1] * f_fₚ .* abs.(gc.x) / sqrt(π * time[i]) .* exp.(-x[n+1]^2 * gc.x.^2 / time[i]))
                J[1:n+1, 1:n] .= abs(x[n+1]) * Δp_σ₀ * (df_fₚ / Δp_σ₀ .- df_fₚ .* fluid_pressure(gc.x, x[n+1], time[i])) .- A
                J[:, n+1] .= x[n+1] / abs(x[n+1]) * (Δp_σ₀ * ((f_fₚ .- τ₀_τₚ) / Δp_σ₀ .- f_fₚ .* fluid_pressure(gc.x, x[n+1], time[i])) .- abs(x[n+1]) * Δp_σ₀ * f_fₚ .* dfluid_pressure(gc.x, x[n+1], time[i]))
            end

            # Newton solve
            res = nlsolve(res!, jac!, vcat(F, aᵢ+0.1), method = :newton)

            # Solutions
            r = res.zero[end]
            F = res.zero[1:n]
            δ = GaussChebyshevFracture.u(gc, F)
            a[i] = res.zero[end]
            δ₀[i] = δ[floor(Int, n/2) + 1]
            aᵢ = a[i]

            println("\t Crack length: ", aᵢ)
            println("\t Max slip: ", δ₀[i])
            println("")
        end

        display(lineplot(sqrt.(time), a))
        open("slip-weakening.csv", "w") do io
            write(io, "time,a,slip\n") # write header
            writedlm(io, [time a δ₀], ',')
        end
        # Check if numerical solutions is correct (check λ)
        # @test converged(res) && isapprox(λ, 1.9125140; rtol=1.0e-02)
    end
end
end
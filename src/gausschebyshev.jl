struct GaussChebyshev{T<:Real}
    " weigth function"
    wf::Function

    " number of quadrature points"
    n::Integer

    " points"
    x::Vector{T}

    " quadrature points"
    s::Vector{T}

    " quadrature weigths"
    w::Vector{T}

    " interpolation matrix"
    L::Matrix{T}

    # " differentiation matrix"
    # D::Matrix{T}

    " integration matrix"
    S::Matrix{T}

    # " derivation matrix"
    # D::Matrix{T}

    " Interpolation matrix"
    function interpolation_matrix(x::Vector{T}, s::Vector{T}; kind::Integer=1)::Matrix{T} where {T<:Real}
        if kind == 1
            ϕ = [cos(k * acos(x[i])) for i in eachindex(x), k in eachindex(s).-1]
            B = [2 * cos(k * acos(s[j])) / length(s) for k in eachindex(x), j in eachindex(s)]
            B = vcat(fill(1 / length(s), length(s))', B)
        elseif kind == 2
            ϕ = [sin((k + 1) * acos(x[i])) / sin(acos(x[i])) for i in eachindex(x), k in eachindex(s).-1]
            B = [2 * sin(acos(s[j])) * sin((k + 1) * acos(s[j])) / (length(s) + 1) for k in eachindex(s).-1, j in eachindex(s)]
        else
            throw(ArgumentError("Chebyshev kind should be 1 or 2 (3 and 4 not implemented yet)!"))
        end

        return ϕ * B
    end

    " Integration matrix"
    function integration_matrix(x::Vector{T}, s::Vector{T}; kind::Integer=2)::Matrix{T} where {T<:Real}
        if kind == 1
            Φ = [-sin(k * acos(x[i])) / k for i in eachindex(x), k in eachindex(x)]
            Φ = hcat(-acos.(x), Φ)
            B = [2 * cos(k * acos(s[j])) / length(s) for k in eachindex(x), j in eachindex(s)]
            B = vcat(fill(1 / length(s), length(s))', B)
        elseif kind == 2
            Φ = [-0.5 * (sin(k * acos(x[i])) / k - sin((k + 2) * acos(x[i])) / (K + 2)) for i in eachindex(x), k in eachindex(x)]
            Φ = hcat(-0.5 * (acos.(x) .- sin.(2 * acos.(x)) / 2), Φ)
            B = [2 * sin(acos(s[j])) * sin((k + 1) * acos(s[j])) / (length(s) + 1) for k in eachindex(s).-1, j in eachindex(s)]
        else
            throw(ArgumentError("Chebyshev kind should be 1 or 2 (3 and 4 not implemented yet)!"))
        end
        return Φ * B
    end
    " Constructor"
    function GaussChebyshev(n::Int64, kind::Integer, T::Type=Float64)
        if kind == 1
            x = [cos(k * π / n) for k = n-1:-1:1]
            s = [cos((2 * k - 1) * π / (2 * n)) for k = n:-1:1]
            return new{T}(x -> 1 / sqrt(1 - x^2), 
                n,
                x,
                s,
                fill(π / n, n),
                interpolation_matrix(x, s; kind=1),
                integration_matrix(x, s; kind=1),
                )
        elseif kind ==2
            x = [cos((2 * k - 1) * π / (2 * (n + 1))) for k = n:-1:1]
            s = [cos(k * π / (n + 1)) for k = n:-1:1]
            return new{T}(x -> sqrt(1 - x^2), 
                n,
                x,
                s,
                [π / (n + 1) * sin(k / (n + 1) * π)^2 for k = n:-1:1],
                interpolation_matrix(x, s, kind=2),
                integration_matrix(x, s, kind=2),
                )
        else
            throw(ArgumentError("Chebyshev kind should be 1 or 2 (3 and 4 not implemented yet)!"))
        end
    end
end

" Returns the solution u"
function u(gc::GaussChebyshev{T}, F::Vector{T})::Vector{T} where {T<:Real}
    return gc.S * F
end

" Returns the gradient of the solution ∇u"
function ∇u(gc::GaussChebyshev{T}, F::Vector{T})::Vector{T} where {T<:Real}
    return gc.wf.(gc.x) .* (gc.L * F)
end
module Func
    include("./setup.jl")
    using .Const

    function sigmoid(x)

        return 1 ./ (exp.(-x) .+ 1.0)
    end

    function diff_sigmoid(x)

        return sigmoid(x) .* (1.0 - sigmoid(x))
    end

    function translate(t)

        return Const.dimB * Const.ω / (exp(Const.ω / t) + 1.0)
    end

    function retranslate(ϵ)

        return Const.ω / log(Const.dimB * Const.ω / ϵ - 1.0)
    end

    function updateB(z)

        n = ones(Float32, Const.dimB)
        prob = 1.0 ./ (1.0 .+ exp.(z))
        pzero = rand(Float32, Const.dimB)
        for ix in 1:Const.dimB
            if pzero[ix] < prob[ix]
                n[ix] = 0.0
            end
        end
        return n
    end

    function updateS(z)

        s = -ones(Float32, Const.dimS)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pup = rand(Float32, Const.dimS)
        for ix in 1:Const.dimS
            if pup[ix] < prob[ix]
                s[ix] = 1.0
            end
        end
        return s
    end

    function energyB(n)
 
        return Const.ω * sum(n)
    end

    function energyS(s)

        sum = 0.0
        for ix in 1:2:Const.dimS-1
            sum += s[ix] * s[ix + 1]
        end
        return -Const.J * sum
    end

    function hamiltonian(n, s)

        w = ones(Const.dimB, Const.dimS)
        sumS = 0.0
        for ix in 1:2:Const.dimS-1
            sumS += s[ix] * s[ix + 1]
        end
        energyS = -Const.J * sumS
 
        return energyB(n) + energyS + Const.δ * transpose(n) * w * s
    end

    function squarehamiltonian(n, s)

        h = hamiltonian(n, s)
        return h^2
    end
end

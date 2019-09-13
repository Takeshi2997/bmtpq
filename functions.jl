module Func
    include("./setup.jl")
    using .Const

    function sigmoid(x)

        return 1 ./ (exp.(-x) .+ 1.0)
    end

    function diff_sigmoid(x)

        return sigmoid(x) .* (1.0 - sigmoid(x))
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

        return prod(s)
    end

    function hamiltonian(n, s)

        return (energyB(n) + energyS(s)) / (Const.dimB + Const.dimS)
    end

    function squarehamiltonian(n, s)

        if energyS(s) < 0
            c = (1.0 / 2.0 / (Const.dimB + Const.dimS))^2
        else
            c = 0.0
        end
        h = hamiltonian(n, s)
        return h^2 + c
    end
end

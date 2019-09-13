module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function forward(network)

        (weight, biasB, biasS, η) = network
        s = ones(Float32, Const.dimS)
        energy = 0.0
        squareenergy = 0.0
        for i in 1:Const.iters_num+Const.burnintime
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            if i > Const.burnintime
                energy += Func.hamiltonian(n, s) / Const.iters_num
                squareenergy += energy^2
            end
        end
        dispersion = squareenergy - energy^2
        return energy, dispersion
    end

    function diff_error(network, ϵ)

        (weight, biasB, biasS, η) = network
        s = ones(Float32, Const.dimS)
        energy = 0.0
        energyB = 0.0
        squareenergy = 0.0
        dweight_h2 = zeros(Float32, Const.dimB, Const.dimS)
        dweight_h = zeros(Float32, Const.dimB, Const.dimS)
        dweight = zeros(Float32, Const.dimB, Const.dimS)
        dbiasB_h2 = zeros(Float32, Const.dimB)
        dbiasB_h = zeros(Float32, Const.dimB)
        dbiasB = zeros(Float32, Const.dimB)
        dbiasS_h2 = zeros(Float32, Const.dimS)
        dbiasS_h = zeros(Float32, Const.dimS)
        dbiasS = zeros(Float32, Const.dimS)

        for i in 1:Const.iters_num+Const.burnintime
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            if i > Const.burnintime
                e = Func.hamiltonian(n, s)
                e2 = Func.squarehamiltonian(n, s)
                energy += e
                energyB += Func.energyB(n)
                squareenergy += e2
                dweight_h2 +=  transpose(s) .* n .* e2
                dweight_h +=  transpose(s) .* n .* e
                dweight +=  transpose(s) .* n
                dbiasB_h2 += n * e2
                dbiasB_h += n * e
                dbiasB += n
                dbiasS_h2 += s * e2
                dbiasS_h += s * e
                dbiasS += s
            end
        end
        energy /= Const.iters_num
        energyB /= Const.iters_num
        squareenergy /= Const.iters_num
        dweight_h2 /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight /= Const.iters_num
        dbiasB_h2 /= Const.iters_num
        dbiasB_h /= Const.iters_num
        dbiasB /= Const.iters_num
        dbiasS_h2 /= Const.iters_num
        dbiasS_h /= Const.iters_num
        dbiasS /= Const.iters_num
        dispersion = squareenergy - energy^2
        error = dispersion - η * (energy - ϵ)^2 / 2.0

        diff_weight = dweight_h2 - squareenergy * dweight_h -
        ((2.0 + η) * energy - η * ϵ) * (dweight_h - energy * dweight)
        diff_biasB = dbiasB_h2 - squareenergy * dbiasB_h - 
        ((2.0 + η) * energy - η * ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS = dbiasS_h2 - squareenergy * dbiasS_h - 
        ((2.0 + η) * energy - η * ϵ) * (dbiasS_h - energy * dbiasS)
        diff_η = -(energy - ϵ)^2 / 2.0

        return error, energy, energyB, dispersion, 
        diff_weight, diff_biasB, diff_biasS, diff_η
    end
end

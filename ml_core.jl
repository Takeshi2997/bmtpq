module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(network, ϵ)

        (weight, biasB, biasS) = network
        n = zeros(Const.dimB)
        energy = 0.0
        energyS = 0.0
        energyB = 0.0
        squareenergy = 0.0
        dweight_h = zeros(Float32, Const.dimB, Const.dimS)
        dweight = zeros(Float32, Const.dimB, Const.dimS)
        dbiasB_h = zeros(Float32, Const.dimB)
        dbiasB = zeros(Float32, Const.dimB)
        dbiasS_h = zeros(Float32, Const.dimS)
        dbiasS = zeros(Float32, Const.dimS)

        for i in 1:Const.iters_num+Const.burnintime
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            if i > Const.burnintime
                e = Func.hamiltonian(n, s)
                e2 = Func.squarehamiltonian(n, s)
                energy += e
                energyS += Func.energyS(s)
                energyB += Func.energyB(n)
                squareenergy += e2
                dweight_h +=  transpose(s) .* n .* e
                dweight +=  transpose(s) .* n
                dbiasB_h += n * e
                dbiasB += n
                dbiasS_h += s * e
                dbiasS += s
            end
        end
        energy /= Const.iters_num
        energyS /= Const.iters_num
        energyB /= Const.iters_num
        squareenergy /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight /= Const.iters_num
        dbiasB_h /= Const.iters_num
        dbiasB /= Const.iters_num
        dbiasS_h /= Const.iters_num
        dbiasS /= Const.iters_num
        dispersion = squareenergy - energy^2
        error = (energy - ϵ)^2

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasB = 2.0 * (energy - ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS = 2.0 * (energy - ϵ) * (dbiasS_h - energy * dbiasS)

#        diff_weight = (dweight_h2 - squareenergy * dweight_h) + 
#        2.0 * ((η - 1.0) * energy - ϵ * η) * (dweight_h - energy * dweight)
#        diff_biasB = η * (dbiasB_h2 - squareenergy * dbiasB_h) + 
#        2.0 * ((η - 1.0) * energy - ϵ * η) * (dbiasB_h - energy * dbiasB)
#        diff_biasS = (dbiasS_h2 - squareenergy * dbiasS_h) + 
#        2.0 * ((η - 1.0) * energy - ϵ * η) * (dbiasS_h - energy * dbiasS)
#        diff_η = (energy - ϵ)^2


        return error, energy, energyS, energyB, dispersion, 
        diff_weight, diff_biasB, diff_biasS
    end

    function forward(network)

        (weight, biasB, biasS, η) = network
        s = [1.0, 1.0]
        energyS = 0.0
        energyB = 0.0
        num = 10000

        for i in 1:num+Const.burnintime
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            if i > Const.burnintime
                energyS += Func.energyS(s)
                energyB += Func.energyB(n)
            end
        end
        energyS /= num
        energyB /= num

        return energyS, energyB
    end
end

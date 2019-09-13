include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

# Make data file
dirname = "./data"
filename = dirname * "/param.dat"
subfilename = dirname * "/subparam.dat"
rm(dirname, force=true, recursive=true)
mkdir(dirname)

# Initialize weight, bias and η
weight = Init.weight(Const.dimB, Const.dimS)
wmoment = zeros(Float32, Const.dimB, Const.dimS)
wvelocity = zeros(Float32, Const.dimB, Const.dimS)
biasB = Init.bias(Const.dimB)
bmomentB = zeros(Float32, Const.dimB)
bvelocityB = zeros(Float32, Const.dimB)
biasS = Init.bias(Const.dimS)
bmomentS = zeros(Float32, Const.dimS)
bvelocityS = zeros(Float32, Const.dimS)
η = -1.0
ηm = 0.0
ηv = 0.0
ϵ = 1.0 / (Const.dimB + Const.dimS)

# Define network
network = (weight, biasB, biasS, η)

# Reed file
# network = open(deserialize, filename)

# Learning
f = open("error.txt", "w")
for it in 1:Const.it_num
    error, energy, energyB, dispersion, dweight, dbiasB, 
    dbiasS, dη = MLcore.diff_error(network, ϵ)

    if (it - 1) % 100 == 0 
        write(f, string(it))
        write(f, "\t")
        write(f, string(error))
        write(f, "\t")
        write(f, string(energy * (Const.dimB + Const.dimS)))
        write(f, "\t")
        write(f, string(energyB))
        write(f, "\t")
        write(f, string(dispersion))
        write(f, "\n")
    end

    # SGD
#    global weight += Const.lr * dweight
#    global biasB += Const.lr * dbiasB
#    global biasS += Const.lr * dbiasS
#    global η += Const.lr * dη

    # Momentum
#    global wmoment = 0.9 * wmoment - Const.lr * dweight
#    global weight += wmoment
#    global bmomentB = 0.9 * bmomentB - Const.lr * dbiasB
#    global biasB += bmomentB
#    global bmomentS = 0.9 * bmomentS - Const.lr * dbiasS
#    global biasS += bmomentS
#    global ηm = 0.9 * ηm - Const.lr * dη
#    global η += ηm 

    # Adam
    lr_t = Const.lr * sqrt(1.0 - 0.999^it) / (1.0 - 0.9^it)
    global wmoment += (1 - 0.9) * (dweight - wmoment)
    global wvelocity += (1 - 0.999) * (dweight.^2 - wvelocity)
    global weight -= lr_t * wmoment ./ (sqrt.(wvelocity) .+ 1.0 * 10^(-7))
    global bmomentB += (1 - 0.9) * (dbiasB - bmomentB)
    global bvelocityB += (1 - 0.999) * (dbiasB.^2 - bvelocityB)
    global biasB -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
    global bmomentS += (1 - 0.9) * (dbiasS - bmomentS)
    global bvelocityS += (1 - 0.999) * (dbiasS.^2 - bvelocityS)
    global biasS -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))
    global ηm += (1 - 0.9) * (dη - ηm)
    global ηv += (1 - 0.999) * (dη.^2 - ηv)
    global η -= lr_t * ηm ./ (sqrt.(ηv) .+ 1.0 * 10^(-7))

    global network = (weight, biasB, biasS, η)
end

close(f)

subparam = (wmoment, wvelocity, bmomentB, bvelocityB,
            bmomentS, bvelocityS, ηm, ηv)

open(io -> serialize(io, network), filename, "w")
open(io -> serialize(io, subparam), subfilename, "w")


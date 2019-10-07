include("./setup.jl")
using .Const, LinearAlgebra, Serialization

function func(t)

    return Const.dimB * Const.ω / (exp(Const.ω / t) + 1.0)
end

f = open("energy-temperature.txt", "w")
for it in 1:1000
    t = it * 0.01
    write(f, string(t))
    write(f, "\t")
    write(f, string(func(t)))
    write(f, "\n")
end
close(f)


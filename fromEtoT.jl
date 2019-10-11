include("./setup.jl")
include("./functions.jl")

using .Const, .Func, LinearAlgebra, Serialization

f = open("energy-temperature.txt", "w")
for iϵ in 1:600
    ϵ = iϵ * 0.0001
    write(f, string(ϵ))
    write(f, "\t")
    write(f, string(Func.retranslate(ϵ)))
    write(f, "\t")
    write(f, string(-Const.J * tanh(Const.J / Func.retranslate(ϵ))))
    write(f, "\n")
end
close(f)


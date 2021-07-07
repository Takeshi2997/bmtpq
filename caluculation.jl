include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

dirname = "./data"

f = open("energy_data.txt", "w")
for itemperature in 1:100

    filename = dirname * "/param_at_" * lpad(itemperature, 3, "0") * ".dat"
    network = open(deserialize, filename)

    energyS, energyB = MLcore.forward(network)
    temperature = Func.retranslate(energyB)

    # Write energy
    write(f, string(temperature))
    write(f, "\t")
    write(f, string(energyS))
    write(f, "\t")
    write(f, string(energyB))
    write(f, "\t")
    write(f, string(-tanh(1.0/temperature)))
    write(f, "\n")
end


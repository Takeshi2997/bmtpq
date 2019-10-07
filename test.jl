include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

dirname = "./data"

for itemperature in 1:100

    temperature = 0.01 * itemperature
    filename = dirname * "/param_at_" * lpad(itemperature, 3, "0") *".dat"
   
    println(filename)
end

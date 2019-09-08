include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

n = [1 0 0 1 1 1 0 0 1]
s = [-0.5 0.5]
e = Func.hamiltonian(n, s)
h2 = Func.squarehamiltonian(n, s)
println(e^2)
println(h2)

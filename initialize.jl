module Init
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function weight(i, j)

        return -ones(Float32, i, j)
    end

    function bias(i)

        return zeros(Float32, i)
    end
end

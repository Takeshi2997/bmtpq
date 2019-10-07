module Const

    struct Param

        # System Size
        dimB::Int64
        dimS::Int64

        # System Param
        ω::Float32

        # Repeat Number
        burnintime::Int64
        iters_num::Int64
        it_num::Int64

        # Learning Rate
        lr::Float32

    end

    # System Size
    dimB = 100
    dimS = 2

    # System Param
    ω = 0.1

    # Repeat Number
    burnintime = 100
    iters_num = 200
    it_num = 5000

    # Learning Rate
    lr = 1.0 * 10^(-2)
end

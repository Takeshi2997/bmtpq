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
    dimB = 500
    dimS = 2

    # System Param
    ω = 0.1

    # Repeat Number
    burnintime = 10
    iters_num = 200
    it_num = 20000

    # Learning Rate
    lr = 1.0 * 10^(-3)
end

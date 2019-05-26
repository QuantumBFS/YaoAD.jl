module YaoAD

using Zygote
using Zygote: gradient, @adjoint, @adjoint!, grad_mut
using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis
import Zygote: Context

using Yao
using YaoBlocks: ConstGate
import Yao: apply!, ArrayReg, statevec, RotationGate

using LuxurySparse, SparseArrays, LinearAlgebra
using BitBasis: controller, controldo, IterControl
using TupleTools

export gradient_check, projection, collect_gradients

"""A safe way of checking gradients"""
function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@adjoint! function dispatch!(op, circuit, params)
    y = dispatch!(op, circuit, params)
    y,
    function (adjy)
        dstk = grad_mut(__context__, params)
        @show dstk, adjy
        @show grad_mut(__context__, y)
        @show collect_gradients(adjy)
        dstk .+= collect_gradients(adjy, empty(params))
        (nothing, nothing, dstk)
    end
end

include("adjbase.jl")
include("adjmat.jl")
include("adjapply.jl")

end # module

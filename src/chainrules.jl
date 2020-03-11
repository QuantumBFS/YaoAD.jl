using ChainRulesCore
using YaoBlocks
using YaoArrayRegister
using YaoBlocks.AD

function ChainRulesCore.rrule(::typeof(YaoBlocks.expect), op::AbstractBlock, reg::ArrayReg, circuit::AbstractBlock)
    out = copy(reg) |> circuit
    ext = expect(op, out)

    function expect_pullback(Δ)
        ∇out = copy(out) |> op
        (in, ∇in), ∇ps = apply_back((out, ∇out), circuit)
    
        return nothing, ∇in, @. ∇ps * 2 * Δ
    end

    return ext, expect_pullback
end

function ChainRulesCore.rrule(::typeof(apply!), reg::ArrayReg, block::AbstractBlock)
    out = apply!(reg, block)
    out, function apply_pullback(outδ)
        (in, inδ), paramsδ = apply_back((out, outδ), block)
        return (inδ, paramsδ)
    end
end

function ChainRulesCore.rrule(::typeof(Matrix), block::AbstractBlock)
    out = Matrix(block)
    out, function matrix_constructor_pullback(outδ)
        paramsδ = mat_back(block, outδ)
        return (paramsδ,)
    end
end

function ChainRulesCore.rrule(::typeof(ArrayReg{B}), raw::AbstractArray) where B
    ArrayReg{B}(raw), adjy->(reshape(adjy.state, size(raw)),)
end

function ChainRulesCore.rrule(::typeof(ArrayReg{B}), raw::ArrayReg) where B
    ArrayReg{B}(raw), adjy->(adjy,)
end

function ChainRulesCore.rrule(::typeof(ArrayReg), raw::AbstractArray)
    ArrayReg(raw), adjy->(reshape(adjy.state, size(raw)),)
end

function ChainRulesCore.rrule(::typeof(copy), reg::ArrayReg)
    copy(reg), adjy->(adjy,)
end

ChainRulesCore.rrule(::typeof(state), reg::ArrayReg) = state(reg), adjy->(ArrayReg(adjy),)
ChainRulesCore.rrule(::typeof(statevec), reg::ArrayReg) = statevec(reg), adjy->(ArrayReg(adjy),)
ChainRulesCore.rrule(::typeof(state), reg::AdjointArrayReg) = state(reg), adjy->(ArrayReg(adjy')',)
ChainRulesCore.rrule(::typeof(statevec), reg::AdjointArrayReg) = statevec(reg), adjy->(ArrayReg(adjy')',)
ChainRulesCore.rrule(::typeof(parent), reg::AdjointArrayReg) = parent(reg), adjy->(adjy',)
ChainRulesCore.rrule(::typeof(Base.adjoint), reg::ArrayReg) = Base.adjoint(reg), adjy->(parent(adjy),)


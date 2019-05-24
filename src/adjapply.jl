include("YaoAD.jl")
using Main.YaoAD
using Yao
using Zygote: @adjoint, grad_mut, @adjoint!, @code_adjoint
const CphaseGate{N, T} = ControlBlock{N,<:ShiftGate{T},<:Any}
const Rotor{N, T} = Union{RotationGate{N, T}, PutBlock{N, <:Any, <:RotationGate{<:Any, T}}, CphaseGate{N, T}}

"""
    generator(rot::Rotor) -> AbstractBlock

Return the generator of rotation block.
"""
generator(rot::RotationGate) = rot.block
generator(rot::PutBlock{N, C, GT}) where {N, C, GT<:RotationGate} = PutBlock{N}(generator(rot|>content), rot |> occupied_locs)
generator(c::CphaseGate{N}) where N = ControlBlock(N, c.ctrol_locs, ctrl_config, control(2,1,2=>Z), c.locs)

@adjoint function ArrayReg{B}(raw::AbstractArray) where B
    ArrayReg{B}(raw), adjy->(adjy.state,)
end

@adjoint function ArrayReg{B}(raw::ArrayReg) where B
    ArrayReg{B}(raw), adjy->(adjy,)
end

@adjoint function copy(reg::ArrayReg) where B
    copy(reg), adjy->(adjy,)
end

@adjoint state(reg::ArrayReg) = state(reg), adjy->(ArrayReg(adjy),)
@adjoint statevec(reg::ArrayReg) = statevec(reg), adjy->(ArrayReg(adjy),)
@adjoint state(reg::AdjointArrayReg) = state(reg), adjy->(ArrayReg(adjy')',)
@adjoint statevec(reg::AdjointArrayReg) = statevec(reg), adjy->(ArrayReg(adjy')',)
@adjoint parent(reg::AdjointArrayReg) = parent(reg), adjy->(adjy',)
@adjoint Base.adjoint(reg::ArrayReg) = Base.adjoint(reg), adjy->(parent(adjy),)

using Zygote
function Zygote.grad_mut(reg::ArrayReg)
    state = zero(reg.state)
    state[1] = 1
    ArrayReg(state)
end

@adjoint! function apply!(reg::AbstractRegister, block)
    x = copy(reg)
    apply!(reg, block),
    function (adjy)
        grad_mut(__context__, reg).state = adjy.state
        adjapply!(x, adjy, block)
    end
end

function adjapply!(x, adjy::AbstractRegister, block::Rotor{N, T}) where {N, T}
    apply!(adjy, block')
    #apply!(y, block')
    dx = projection(T(0), im/2*((x |> generator(block))' * adjy))
    (adjy, dx)
end

function adjapply!(x, adjy::AbstractRegister, block::Union{ConstantGate, PutBlock{<:Any, <:Any, <:ConstantGate}, ControlBlock{<:Any, <:ConstantGate}}) where {N, T}
    apply!(adjy, block')
    #apply!(y, block')
    (adjy, nothing)
end

@adjoint! function dispatch!(circuit, params)
    dispatch!(circuit, params),
    function (adjy)
        dstk = grad_mut(__context__, params)
        dstk .+= collect_gradients(adjy)
        (nothing, dstk)
    end
end

@inline function adjunrows!(adjy, adjU, state::AbstractVector, inds::AbstractVector, U::AbstractMatrix)
    @inbounds state[inds] = U*state[inds]
    state
    outer_projection(U, state_in, state_out)
end

@inline function adjunrows!(adjy, adjU, state::AbstractMatrix, inds::AbstractVector, U::AbstractMatrix)
    @inbounds for k in 1:size(state, 2)
        state[inds, k] = U*state[inds, k]
    end
    state
end

using YaoArrayRegister: _prepare_instruct, _instruct!
@adjoint _prepare_instruct(args...) = _prepair_instruct(args...), adjy->nothing
@adjoint! function _instruct!(state::AbstractVecOrMat{T}, U::AbstractMatrix{T}, locs_raw::SVector, ic::IterControl) where T
    x = copy(state)
    _instruct!(state, U, locs_raw, ic),
    function (adjy)
        controldo(ic) do i
            adjunrows!(adjy, adjU, x, locs_raw .+ i, U)
        end
        (adjy, adjU, nothing, nothing)
    end
end

using Test, Random
ng(f, θ, δ=1e-5) = (f(θ+δ/2) - f(θ-δ/2))/δ

@testset "apply! rotor" begin
    nbit = 1
    θ = 0.5
    # Chain Block
    Random.seed!(4)

    reg = rand_state(1)
    h = Z
    f(x) = expect(h, apply!(copy(reg), Rx(x))) |> real
    #display(@code_adjoint f(θ))
    @test isapprox(f'(θ), ng(f, θ), atol=1e-4)

    g(x) = expect(h, apply!(apply!(copy(reg), Rx(x)), Ry(x))) |> real
    @test isapprox(g'(θ), ng(g, θ), atol=1e-4)

    nbit = 5
    operator = rand_unitary(2)
    b = randn(ComplexF64, 1<<nbit)
    h(operator) = abs(b'*instruct!(b, operator, (2,), (), ()))
    @show h(operator)
    @test gradient_check(h, operator)
end

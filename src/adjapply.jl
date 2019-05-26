const CphaseGate{N, T} = ControlBlock{N,<:ShiftGate{T},<:Any}
const Rotor{N, T} = Union{RotationGate{N, T}, PutBlock{N, <:Any, <:RotationGate{<:Any, T}}, CphaseGate{N, T}}

dagger_map(G::Union{Val{:X}, Val{:Y}, Val{:Z}, Val{:H}, Val{:P0}, Val{:P1}}) = G
dagger_map(G::Val{:T}) = Val(:Tdag)
dagger_map(G::Val{:Tdag}) = Val(:T)
dagger_map(G::Val{:Sdag}) = Val(:S)
dagger_map(G::Val{:Pu}) = Val(:Pd)
dagger_map(G::Val{:Pd}) = Val(:Pu)

"""
    generator(rot::Rotor) -> AbstractBlock

Return the generator of rotation block.
"""
generator(rot::RotationGate) = rot.block
generator(rot::PutBlock{N, C, GT}) where {N, C, GT<:RotationGate} = PutBlock{N}(generator(rot|>content), rot |> occupied_locs)
generator(c::CphaseGate{N}) where N = ControlBlock(N, c.ctrol_locs, ctrl_config, control(2,1,2=>Z), c.locs)

@adjoint function ArrayReg{B}(raw::AbstractArray) where B
    ArrayReg{B}(raw), adjy->(reshape(adjy.state, size(raw)),)
end

@adjoint function ArrayReg{B}(raw::ArrayReg) where B
    ArrayReg{B}(raw), adjy->(adjy,)
end

@adjoint function ArrayReg(raw::AbstractArray)
    ArrayReg(raw), adjy->(reshape(adjy.state, size(raw)),)
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
Zygote.grad_mut(reg::ArrayReg) = Ref{Any}(ArrayReg(zero(reg.state)))

const NoDiffGate = Union{ConstantGate{N}, PutBlock{N, <:Any, <:ConstantGate}, ControlBlock{N, <:ConstantGate}} where N

@adjoint! function apply!(reg::AbstractRegister, block::Union{NoDiffGate, Rotor})
    x = copy(reg)
    apply!(reg, block),
    function (adjy)
        grad_mut(__context__, reg).x = adjy
        adjapply!(x, adjy, block)
    end
end

function adjapply!(x, adjy::AbstractRegister, block::Rotor{N, T}) where {N, T}
    apply!(adjy, block')
    #apply!(y, block')
    dx = projection(T(0), im/2*((x |> generator(block))' * adjy))
    (adjy, dx)
end

function adjapply!(x, adjy::AbstractRegister, block::NoDiffGate)
    apply!(adjy, block')
    #apply!(y, block')
    (adjy, nothing)
end

@inline function adjunrows!(adjy, adjU, state::AbstractVector, inds::AbstractVector, U::AbstractMatrix)
    @inbounds outer_projection!(adjU, adjy[inds], state[inds]')
    @inbounds adjy[inds] = U'*adjy[inds]
end

@inline function adjunrows!(adjy, adjU, state::AbstractMatrix, inds::AbstractVector, U::AbstractMatrix)
    @inbounds for k in 1:size(state, 2)
        @inbounds outer_projection!(adjU, adjy[inds, k], state[inds, k]')
        @inbounds adjy[inds, k] = U'*adjy[inds, k]
    end
end

@inline adjunrows!(adjy, adjU, state::AbstractVector, inds::AbstractVector, U::IMatrix) = nothing

using YaoArrayRegister: _prepare_instruct, _instruct!, autostatic, sort_unitary
using YaoBlocks: _check_size
@adjoint autostatic(A; kwargs...) = autostatic(A; kwargs...), adjy->(adjy,)
@adjoint function sort_unitary(U::AbstractMatrix, locs::NTuple{N, Int}) where N
    sort_unitary(U, locs), adjy->(sort_unitary(adjy, locs|>TupleTools.sortperm), nothing)
end
@adjoint _check_size(args...) = _check_size(args...), adjy->nothing
@adjoint _prepare_instruct(args...) = _prepare_instruct(args...), adjy->nothing
@adjoint! function _instruct!(state::AbstractVecOrMat{T}, U::AbstractMatrix{T}, locs_raw, ic::IterControl) where T
    x = copy(state)
    _instruct!(state, U, locs_raw, ic),
    function (adjy)
        adjU = _render_adjU(U)
        controldo(ic) do i
            adjunrows!(adjy, adjU, x, locs_raw .+ i, U)
        end
        #grad_mut(__context__, state) .= adjy
        (adjy, adjU, nothing, nothing)
    end
end

#=  do we really need this?
@adjoint! function instruct!(state::AbstractVecOrMat{T}, symbol::Val, locs::NTuple{N, Int}) where {T, N}
    instruct!(state, symbol, locs),
    function (adjy)
        instruct!(adjy, dagger_map(symbol), locs),
        (adjy, nothing, nothing)
    end
end

@adjoint! function instruct!(
    state::AbstractVecOrMat{T}, symbol::Val,
    locs::NTuple{N1, Int},
    control_locs::NTuple{N2, Int},
    control_bits::NTuple{N3, Int}) where {T, N1, N2, N3}

    instruct!(state, symbol, locs),
    function (adjy)
        instruct!(adjy, dagger_map(symbol), locs, control_locs, control_bits),
        (adjy, nothing, nothing, nothing, nothing)
    end
end
=#

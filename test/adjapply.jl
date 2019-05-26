using YaoAD
using Yao, Zygote
using Test, Random
using LinearAlgebra, SparseArrays, LuxurySparse

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
end

@testset "apply! general mat" begin
    nbit = 5
    Random.seed!(3)

    h(b, operator) = sum(abs2, b'*instruct!(copy(b), operator, (2,), (), ()))

    for operator = [rand_unitary(2)]
        for b in [randn(ComplexF64, 1<<nbit), randn(ComplexF64, 1<<nbit, 5)]
            # the gradient of operator
            @test gradient_check(op->h(b, op), operator; η=1e-7)

            # the gradient of wave function
            @test gradient_check(b->h(b, operator), b; η=1e-7)
        end
    end
end

@testset "apply! general mat" begin
    nbit = 5
    Random.seed!(3)

    h(b, operator) = sum(abs2, b'*instruct!(copy(b), operator, (2,), (), ()))

    for operator = [rand_unitary(2), Diagonal([0.4im, 0.6]), SparseMatrixCSC([0.3 0; 0.5im 0]), pmrand(ComplexF64, 2)]
        for b in [randn(ComplexF64, 1<<nbit), randn(ComplexF64, 1<<nbit, 5)]
            # the gradient of operator
            @test gradient_check(op->h(b, op), operator; η=1e-7)

            # the gradient of wave function
            @test gradient_check(b->h(b, operator), b; η=1e-7)
        end
    end
end

@testset "apply! chain" begin
    nbit = 5
    Random.seed!(3)

    #c = chain([control(nbit, 2, 3=>Rx(0.5))])
    c = control(nbit, 2, 3=>Rx(0.5))
    op = put(nbit, 3=>Z)
    psi = rand_state(nbit)
    @test gradient_check(state-> expect(op, ArrayReg(state)) |> real, randn(ComplexF64, 1<<nbit))
    function loss(c)
       #psi = zero_state(nbit) |> c
       expect(c, psi) |> real
    end

    function L(theta)
       dispatch!(c, theta)
       loss(c)
    end

    @show L(0.7)

    @test_broken collect_gradients(loss'(c))[] == ng(L, 0.5)
    #c = chain([put(nbit, 2=>H), control(nbit, 2, 3=>Rx(0.5))])
end

@testset "*, focus, relax" begin
    Random.seed!(10)
    nbit = 5
    st = randn(ComplexF64, 1<<nbit)
    St = randn(ComplexF64, 1<<nbit, 5)
    reg2 = rand_state(ComplexF64, nbit) |> focus!(5,3,2)
    Reg2 = rand_state(ComplexF64, nbit; nbatch=5) |> focus!(5,3,2)
    loss(st, reg) = sum(abs2, (focus!(ArrayReg(copy(st)), (4,2,3)) |> relax!(4,3; to_nactive=4) |> focus!(2,4))'*reg)
    @show loss(st, reg2)
    @test gradient_check(st -> loss(st, reg2), st)
    @test gradient_check(st -> loss(st, Reg2), St)
end

using Zygote
using YaoAD, Yao, Random
using Test, LuxurySparse
using SparseArrays

@testset "adjbase" begin
    include("adjbase.jl")
end

@testset "adjapply" begin
    include("adjapply.jl")
end

@testset "adjmat" begin
    include("adjmat.jl")
end

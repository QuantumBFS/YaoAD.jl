# YaoAD

Automatic differentiable quantum circuit.

## Install
```julia
pkg> add git@github.com:QuantumBFS/YaoAD.git
```

## Run an example
```julia
julia examples/vqe.jl
```

## Table of Supported AD Blocks
### Tested `mat` BP
* `KronBlock`
* `ChainBlock`
* `RotationGate`
* `TimeEvolution`
* `PutBlock`
* `GeneralMatrixBlock`

#### TODO-mat

### Tested `apply!` BP

#### TODO-apply!
* PutBlock{<:Any, <:Any, <:RotationGate}
* PutBlock{<:Any, <:Any, <:ConstantGate}
* ChainBlock
* RotationGate

# YaoAD

Automatic differentiable quantum circuit.

## Install
```julia
pkg> add <this repo.git>
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

### Tested `apply!` BP

using Pkg
Pkg.activate("./")

using DLDS
using BenchmarkTools
using LinearAlgebra

num_latents = 8
num_timepoints = 1000
num_motifs = 15

X = randn(num_latents, num_timepoints)
c = randn(num_motifs, num_timepoints - 1)
F = randn(num_motifs, num_latents, num_latents)

T = Float64
FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs)

@benchmark update_c!(
    c,
    FX_prod,
    X,
    F,
    smooth_coeff = T(0.01),
    l1_coeff = T(0.01),
    max_iter = 10,
)

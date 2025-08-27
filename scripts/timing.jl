using Pkg
Pkg.activate("./")

using DLDS
using BenchmarkTools
using LinearAlgebra

num_latents = 3
num_observations = 10
num_timepoints = 1000

D = randn(num_observations, num_latents)

X = randn(num_latents, num_timepoints)

Y = randn(num_observations, num_timepoints)


# Y = D * X

@benchmark fit = update_D!(D, 30.0, X, Y, sign_coeff=0.01, frobenius_coeff=0.01)


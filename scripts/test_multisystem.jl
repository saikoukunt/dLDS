using Distributed
addprocs(12 - nprocs())

@everywhere using LinearAlgebra
@everywhere using DLDS
# @everywhere LinearAlgebra.BLAS.set_num_threads(1)
using BenchmarkTools

X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
# X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 15
num_latents = 8
T = Float64

@time F_hat = fit_no_obs_model(
	X,
	num_motifs,
	max_iter = 200,
	F_decorr_coeff = 0.1,
	F_lr_init = 10.0,
	c_l1_coeff = 0.4,
	c_smooth_coeff = 0.4,
	num_snippets = 10,
)

c_hat = Vector{Matrix{Float64}}(undef, 50)
for trial in axes(c, 1)
	c_hat[trial] = Matrix{Float64}(undef, num_motifs, 2999)
end

update_c_parallel!(
	c_hat,
	X,
	F_hat,
	smooth_coeff = 0.4,
	l1_coeff = 0.5,
)


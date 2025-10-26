using Distributed
addprocs(12 - nprocs())

@everywhere using LinearAlgebra
@everywhere using DLDS
using BenchmarkTools

X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)

num_motifs = 6
num_latents = 8
num_snippets = 50
T = Float64

c_snips = Vector{Matrix{T}}(undef, num_snippets)
for i in 1:num_snippets
	c_snips[i] = zeros(num_motifs, 200 - 1)
end

F_hat = Vector{Matrix{Float64}}(undef, num_motifs)
for i in 1:num_motifs
	F_hat[i] = init_matrix(
		InitDistribution.Normal(),
		(num_latents, num_latents),
		0)
end

F_old = Vector{Matrix{Float64}}(undef, num_motifs)
for i in 1:num_motifs
	F_old[i] = similar(F_hat[1])
end
gradient_sum = Vector{Matrix{Float64}}(undef, num_motifs)
for i in 1:num_motifs
	gradient_sum[i] = similar(F_hat[1])
end
temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

for i in 1:100
	for i in 1:num_motifs
		F_old[i] .= F_hat[i]
	end
	X_snippets, trial_inds = sample_snippets(X, num_snippets, 200)
	for i in axes(X_snippets, 1)
		trial_id, t_start, t_end = trial_inds[i]
		c_snips[i] = c[trial_id][:, t_start:t_end]
	end

	update_F!(
		F_hat,
		gradient_sum,
		temp_gradient,
		x_hat_next,
		update_F_residuals,
		X_snippets,
		c_snips,
		10.0,
		decorr_coeff = 0.1,
	)

	latent_recon_err =
		calculate_latent_recon_error!(x_hat_next, F_hat, X_snippets, c_snips)
	dF = calculate_delta_F(F_hat, F_old)
	println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")
end

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


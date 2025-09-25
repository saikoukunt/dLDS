using Distributed
addprocs(11)

@everywhere using LinearAlgebra
@everywhere using DLDS
# @everywhere LinearAlgebra.BLAS.set_num_threads(1)
using BenchmarkTools

# X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 6
num_latents = 8
T = Float64

F = fit_no_obs_model(X, num_motifs)


X_trial, c_trial = sample_snippets(X, c, 500, 200)
# c_hat = Vector{Matrix{T}}(undef, 50)
# for tr in axes(c_hat, 1)
#     c_hat[tr] = zeros(num_motifs, 199)
# end

F_hat::Array{T,3} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_latents, num_latents),
        0,
    )

gradient_sum::Array{T,3} = similar(F)                          # for update f! 
temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

@benchmark(
    update_F!(
        $F_hat,
        $gradient_sum,
        $temp_gradient,
        $x_hat_next,
        $update_F_residuals,
        $X_trial,
        $c_trial,
        1.0,
    )
)

@benchmark(
    update_F_parallel!(
        $F_hat,
        $X_trial,
        $c_trial,
        1.0
    )
)

@benchmark(
    update_c_parallel!(
        $c_hat,
        $X_trial,
        $F,
        smooth_coeff = 0.2,
        l1_coeff = 0.2,
        max_iter = 3000,
        tol = 1e-8,
        warm_start = false,
    )
)

for i in 1:750
    update_c_parallel!(
        c_hat,
        X_trial,
        F,
        smooth_coeff = 0.2,
        l1_coeff = 0.2,
        max_iter = 3000,
        tol = 1e-8,
        warm_start = false,
    )

    latent_recon_err =
        calculate_latent_recon_error!(x_hat_next, F, X_trial[1], c_hat[1])
    println("Iter $(i) - Recon Err: $(latent_recon_err)")
end

plot_cs(c_hat[15], c_trial[15])
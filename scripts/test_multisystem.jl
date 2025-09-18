using DLDS

# X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 6
num_latents = 8
T = Float64

X_trial, c_trial = sample_snippets(X, c, 50, 200)
c_hat = Vector{Matrix{T}}(undef, 50)
c_hat[1] = zeros(num_motifs, 199)

x_hat_next = Vector{T}(undef, 8)

for i in 1:750
    trial_data = (
        c_hat[1],
        X_trial[1],
        F,
        (
            smooth_coeff = 0.2,
            l1_coeff = 0.2,
            max_iter = 3000,
            tol = 1e-8,
            warm_start = false,
        ),
    )

    c_hat[1] = worker_update_c(trial_data)

    latent_recon_err =
        calculate_latent_recon_error!(x_hat_next, F, X_trial[1], c_hat[1])
    println("Iter $(i) - Recon Err: $(latent_recon_err)")
end

plot_cs(c_hat[1], c_trial[1])
using DLDS

# X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 6
num_latents = 8
T = Float64

# solve for F and c simultaneously
F_hat, c_hat = fit_no_obs_model(
    @view(X[1, :, :]),
    num_motifs,
    c_l1_coeff = 0.25,
    F_lr_init = 1.0,
    c_smooth_coeff = 0.2,
    max_iter = 750,
)

# solve for c given F
FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs) # for update_c!
FX_prod_gram::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
c_hat = Matrix{Float64}(undef, size(c, 2), size(c, 3) - 1)
update_c!(
    c_hat,
    FX_prod,
    FX_prod_gram,
    X[1, :, :],
    F,
    smooth_coeff = 0.2,
    l1_coeff = 0.2,
)
plot_cs(c[1, :, :], c_hat)

# solve for F given c
gradient_sum::Array{T,3} = similar(F)                          # for update f! 
temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)
F_hat_orig = init_matrix(InitDistribution.Normal(), size(F), 0)

start = time()
F_hat = copy(F_hat_orig)
F_lr = 0.5
trial_id = 1
for i in 1:750
    F_old = copy(F_hat)
    update_F!(
        F_hat,
        gradient_sum,
        temp_gradient,
        x_hat_next,
        update_F_residuals,
        X[trial_id, :, :],
        c[trial_id, :, :],
        F_lr;
        decorr_coeff = 0.2,
    )
    F_lr *= 0.99995

    latent_recon_err = calculate_latent_recon_error!(
        x_hat_next,
        F_hat,
        X[trial_id, :, :],
        c[trial_id, :, :],
    )
    dF = calculate_delta_F(F_hat, F_old)
    println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")
end
plot_Fs(F_hat, F)
print(time() - start)

start = time()
F_hat_matlab = copy(F_hat_orig)
F_lr = 0.5
for i in 1:750
    F_old = copy(F_hat_matlab)
    matlab_update_F!(
        F_hat_matlab,
        X[trial_id, :, :],
        c[trial_id, :, :],
        lr_F = F_lr;
        lambda_F = 0.2,
    )
    F_lr *= 0.99995

    latent_recon_err = calculate_latent_recon_error!(
        x_hat_next,
        F_hat_matlab,
        X[trial_id, :, :],
        c[trial_id, :, :],
    )
    dF = calculate_delta_F(F_hat_matlab, F_old)
    println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")
end
plot_Fs(F_hat_matlab, F)
print(time() - start)
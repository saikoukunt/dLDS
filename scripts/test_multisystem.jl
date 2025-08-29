using DLDS
using Plots

X, c, F = simulate_two_subsystems_no_obs(5000, [4, 4], [3, 3], 50)

F_hat, c_hat, latent_recon_err = fit_no_obs_model(
    X[1, :, :],
    8;
    max_iter = 500,
    c_l1_coeff = 0.5,
    c_smooth_coeff = 0.1,
    c_fista_max_iter = 500,
)

# function fit_no_obs_model(
#     X::AbstractMatrix{T},
#     num_motif::Int;
#     random_seed::Int = 0,
#     max_iter::Int = 3000,
#     c_l1_coeff::T = zero(T),
#     c_l1_coeff_decay::T = T(1),
#     c_smooth_coeff::T = zero(T),
#     c_fista_tol::T = T(1e-8),
#     c_fista_max_iter::Int = 1000,
#     F_normalize_matrix::Bool = true,
#     F_normalize_gradient::Bool = false,
#     F_perturb_threshold::T = T(1e-5),
#     F_noise_sigma::T = T(0.1),
#     F_init_max_corr::T = zero(T),
#     F_lr_decay::T = T(0.999),
#     verbose::Bool = true,
# ) where {T<:AbstractFloat}

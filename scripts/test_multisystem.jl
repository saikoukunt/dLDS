using DLDS
using Plots

X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [1, 1], 50)

# c_hat = infer_no_obs_state(
#     F,
#     @view(X[1, :, :]);
#     c_l1_coeff = 0.2,
#     c_smooth_coeff = 0.1,
#     c_fista_max_iter = 1000,
#     c_fista_tol = Float64(1e-8),
# )

num_motifs = 2
num_latents = 8

T = Float64
F_hat::Array{Float64,3} = init_matrix(InitDistribution.Normal(), (num_motifs, 8, 8), 0)
gradient_sum::Array{T,3} = similar(F_hat)                          # for update f! 
temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

# test F gradient descent alone
lr = 0.1
for i in 1:3000
    F_old = copy(F_hat)
    # update f
    update_F!(
        F_hat,
        gradient_sum,
        temp_gradient,
        x_hat_next,
        update_F_residuals,
        @view(X[1, :, :]),
        @view(c[1, :, :]),
        lr;
        normalize_F = true,
        decorr_coeff = 0.2,
    )

    # calculate and print reconstruction error
    latent_recon_err =
        calculate_latent_recon_error!(x_hat_next, F_hat, X[1, :, :], c[1, :, :])
    dF = calculate_delta_F(F_hat, F_old)
    println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")
end

function plot_cs(c_hat::AbstractMatrix{F}, c::AbstractArray{F}) where {F<:AbstractFloat}
    plot(c[1, 1:3, :]')
    plot!(c[1, 4:6, :]' .+ 4)
    plot!(c_hat[1:3, :]' .+ 2)
    plot!(c_hat[4:6, :]' .+ 6)
end

plot_cs(c_hat, c)

using Distributed
addprocs(10 - nprocs())

@everywhere using LinearAlgebra
@everywhere using DLDS
# @everywhere LinearAlgebra.BLAS.set_num_threads(1)
using BenchmarkTools

# X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 6
num_latents = 8
T = Float64

@time F_hat = fit_no_obs_model(X, num_motifs, max_iter = 3000)

F_hat::Array{T,3} =
    init_matrix(InitDistribution.Normal(), (num_motifs, num_latents, num_latents), 0)

gradient_sum::Array{T,3} = similar(F)                          # for update f! 
temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

for i in 1:750
    X_snippets, indices = sample_snippets(X, 500, 200)
    c_snippets = Vector{Matrix{Float64}}(undef, size(X_snippets, 1))
    for i in axes(X_snippets, 1)
        trial, t_start, t_end = indices[i]
        c_snippets[i] = c[trial][:, t_start:t_end]
    end

    update_F!(
        F_hat,
        gradient_sum,
        temp_gradient,
        x_hat_next,
        update_F_residuals,
        X,
        c,
        1.0,
        decorr_coeff = 0.0,
    )

    latent_recon_err =
        calculate_latent_recon_error!(x_hat_next, F_hat, X_snippets, c_snippets)
    println("Iter $(i) - Recon Err: $(latent_recon_err)")
end

plot_cs(c_hat[15], c_trial[15])

decorr_coeff = 0.05
F_hat_decorr = copy(F_hat)

for t in 1:10
    gradient_sum .= 0
    for i in axes(F_hat_decorr, 1)
        for j in axes(F_hat_decorr, 1)
            if i ≠ j
                # # compute the trace without a new implicit allocation
                # trace = T(0)
                # for k in axes(F, 3)
                #     trace += dot(@view(F[i, :, k]), @view(F[j, :, k]))
                # end

                # # add the decorrelation term to the running gradient
                # axpy!(trace, F[j, :, :], grad_i)
                gradient_sum[i, :, :] +=
                    decorr_coeff *
                    tr(F_hat_decorr[i, :, :]' * F_hat_decorr[j, :, :]) *
                    F_hat_decorr[j, :, :]
            end
        end
    end
    @. F_hat_decorr -= gradient_sum
end

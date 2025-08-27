include("./matrix_utils.jl")

using LinearAlgebra
using Distributions
using Random
using Statistics
using ProximalOperators
using ProximalAlgorithms
using SparseArrays
using Printf
using GLMNet
using LoopVectorization
using Base.Threads

function train_dLDS(
    Y::AbstractMatrix{T},
    num_latents::Int,
    num_motifs::Int;
    random_seed::Int = 0,
    max_iter::Int = 3000,
    recon_threshold::T = T(1e-3),
    x_l1_coeff::T = zero(T),
    c_l1_coeff::T = zero(T),
    c_l1_coeff_decay::T = T(1),
    c_smooth_coeff::T = zero(T),
    D_lr::T = T(30),
    D_sign_coeff::T = zero(T),
    D_frobenius_coeff::T = zero(T),
    F_lr_init::T = T(30),
    F_normalize_matrix::Bool = true,
    F_normalize_gradient::Bool = false,
    F_perturb_threshold::T = T(1e-5),
    F_noise_sigma::T = T(0.1),
    F_init_max_corr::T = zero(T),
    F_lr_decay::T = T(0.8),
    verbose::Bool = true,
) where {T<:AbstractFloat}
    """
    train_dLDS()

    Placeholder function for training a dLDS model.
    """
    println("Training dLDS model...")

    num_observations::Int = size(Y, 1)
    num_timepoints::Int = size(Y, 2)

    # Initialize/pre-allocated model parameters and state
    D::tMatrix{T} = init_matrix(
        InitDistribution.Sparse(),
        (num_observations, num_latents),
        random_seed;
        k = 4,
    )
    F::Array{T,3} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_latents, num_latents),
        random_seed,
    )
    validate_F_separation!(F_init_max_corr)
    c::Matrix{T} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_timepoints - 1),
        random_seed,
    )
    X::Matrix{T} = Matrix{T}(undef, num_latents, num_timepoints)

    recon_err::Vector{T} = zeros(T, max_iter)
    F_lr::T = F_lr_init
    i::Int = 1

    # Pre-allocate for intermediate results
    data_prediction::Matrix{T} = similar(Y)
    FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs) # for update_c!
    gradient_sum::Array{T,3} = similar(F)                          # for update f! 
    temp_gradient::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
    x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
    update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)
    latent_recon_err::Vector{T} = zeros(T, num_timepoints)
    data_recon_err = Inf

    while (data_recon_err > recon_threshold) && (i <= max_iter)
        update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

        c_l1_coeff *= c_l1_coeff_decay
        if i > 1
            update_c!(c, FX_prod, X, F, c_smooth_coeff, c_l1_coeff; warm_start = True)
        end

        update_D!(
            D,
            D_lr,
            X,
            Y;
            sign_coeff = D_sign_coeff,
            frobenius_coeff = D_frobenius_coeff,
        )

        update_F!(
            F,
            gradient_sum,
            temp_gradient,
            x_hat_next,
            update_F_residuals,
            X,
            c,
            F_lr;
            normalize_F = F_normalize_matrix,
            normalize_gradient = F_normalize_gradient,
        )
        F_lr *= F_lr_decay

        data_recon_err = calculate_data_recon_error!(data_prediction, Y, D, X)
        latent_recon_err[i] = calculate_latent_recon_error!(x_hat_next, F, X, c)

        if verbose
            println(
                "Iter $(i): Data Rec. Error: $(data_recon_err), Latent Rec. Error: $(latent_recon_err[i]) ",
            )
        end
        i += 1
    end

    return D, F, X, c, latent_recon_err
end

function calculate_data_recon_error!(
    prediction::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    D::AbstractMatrix{T},
    X::AbstractMatrix{T},
) where {T<:AbstractFloat}
    mul!(prediction, D, X)
    @. prediction .= Y - prediction     # residual, reusing array to avoid extra allocation

    return dot(prediction, prediction) / length(Y)
end

function calculate_latent_recon_error!(
    x_hat_next::AbstractVector{T},
    F::AbstractArray{T,3},
    X::AbstractMatrix{T},
    c::AbstractMatrix{T},
) where {T<:AbstractFloat}
    total_error::T = zero(T)

    for t in 1:size(X, 2)-1
        step_dynamics!(x_hat_next, @view(X[:, t]), @view(c[:, t]), F)
        x_hat_next .= @view(X[:, t+1]) .- x_hat_next    # residual, reusing array to avoid extra allocation

        total_error += dot(x_hat_next, x_hat_next)
    end

    return total_error / (size(x_hat_next, 1) * size(X, 2) - 1)
end

"""
    update_c!(c, FX_prod, X, F, smooth_coeff, l1_coeff; max_iter, tol, warm_start)

Calculates an update estimate of dynamics motif coefficients.

# Arguments

- `c`: Dynamics coefficient history of the system.
- `FX_prod`: A preallocated (# of latents X # of motifs) matrix to hold intermediate results.
- `X`: Latent state history of the system.
- `F`: Dynamics motif matrices.
- `smooth_coeff`: Coefficient for c smoothness penalty.
- `l1_coeff`: Coefficient for c sparsity penalty.
- `max_iter`: Max iterations for FISTA solver.
- `tol`: Stopping tolerance for FISTA solver.
- `warm_start`: Whether to use previous estimates of c_t as initial guess for FISTA solver.
"""
function update_c!(
    c::AbstractMatrix{T},
    FX_prod::AbstractMatrix{T},
    X::AbstractMatrix{T},
    F::AbstractArray{T,3};
    smooth_coeff::T = zero(T),
    l1_coeff::T = zero(T),
    max_iter::Int = 10,
    tol::T = 1e-6,
    warm_start::Bool = false,
) where {T<:AbstractFloat}
    for t in axes(c, 2)
        for i in axes(F, 1)
            mul!(@view(FX_prod[:, i]), @view(F[i, :, :]), @view(X[:, t]))
        end

        reconstruction_loss = LeastSquares(FX_prod, @view(X[:, t+1]))
        if smooth_coeff > 0 && t > 1
            smoothness_penalty = LeastSquares(I, @view(c[:, t-1]), λ = 2 * smooth_coeff)
            f = reconstruction_loss + smoothness_penalty
        else
            f = reconstruction_loss
        end
        l1_penalty = NormL1(l1_coeff)

        solver = ProximalAlgorithms.FastForwardBackward(maxit = max_iter, tol = tol)
        initial_guess = warm_start ? @view(c[:, t]) : zeros(T, size(c, 1))

        solution, iters = solver(initial_guess, f = f, g = l1_penalty)
        @view(c[:, t]) .= solution
    end
end

"""
    update_F!(F, gradient_sum, temp_gradient, x_hat_next, residuals, X, c, lr_F; normalize_gradient, normalize_F)

Updates elements of F via gradient descent.

# Arguments:
- `F`: Dynamics motif matrices.
- `gradient_sum`: Preallocated array (same size as F) for gradient accumulation.
- `temp_gradient`: Preallocated array (# of latents X # of latents) for gradient calculation.
- `x_hat_next`: Preallocated vector (# of latents) for gradient calculation.
- `residuals`: Preallocated vector (# of latents) for gradient calculation.
- `X`: Latent state history of the system.
- `c`: Dynamics coefficient history of the system.
- `lr_F`: Learning rate.
"""
function update_F!(
    F::AbstractArray{T,3},
    gradient_sum::AbstractArray{T,3},
    temp_gradient::AbstractMatrix{T},
    x_hat_next::AbstractVector{T},
    residuals::AbstractVector{T},
    X::AbstractMatrix{T},
    c::AbstractMatrix{T},
    lr_F::T;
    normalize_gradient::Bool = false,
    normalize_F::Bool = true,
) where {T<:AbstractFloat}
    fill!(gradient_sum, zero(T))

    # NOTE: can potentially batch this to make it faster
    # Calculate the sum of the gradients over time w.r.t each F
    for t in 1:size(X, 2)-1
        step_dynamics!(x_hat_next, @view(X[:, t]), @view(c[:, t]), F)
        residuals .= @view(X[:, t+1]) .- x_hat_next
        mul!(temp_gradient, residuals, @view(X[:, t])')

        for i in axes(F, 1)
            axpy!(c[i, t], temp_gradient, @view(gradient_sum[i, :, :]))
        end
    end

    # Take the gradient steps
    for i in axes(F, 1)
        @view(gradient_sum[i, :, :]) ./= (size(X, 2) - 1)
        if normalize_gradient
            normalize_matrix!(@view(gradient_sum[i, :, :]))
        end

        @. @view(F[i, :, :]) -= 2 * lr_F * @view(gradient_sum[i, :, :])
        if normalize_F
            normalize_matrix!(@view(F[i, :, :]))
        end
    end

    nan_indices = isnan.(F)
    F[nan_indices] .= rand(Uniform(0, 1), sum(nan_indices))
end

# TODO: Think about if we should do cv lasso instead
"""
    update_X(X, D, Y, lambda_l1=0.0)

Infers the latent state vector with Lasso regression given the loading matrix D and observations y.

# Arguments
- `D`: Loading matrix.
- `Y`: Observations from the system.
- `lambda_l1`: L1 regularization parameter.

# Returns
- Inferred state vector.
"""
function update_X!(
    X::AbstractMatrix{T},
    D::AbstractMatrix{T},
    Y::AbstractMatrix{T};
    lambda_l1::T = zero(T),
) where {T<:AbstractFloat}
    if iszero(lambda_l1)
        # X .= D \ Y
        X .= pinv(D) * Y
    else
        fit = glmnet(D, Y, MvNormal(), alpha=1.0, lambda=[lambda_l1], intercept=false)
        X .= fit.betas[:,:,1]
    end

    return X
end

"""
    update_D(D, lr_D, x, y; sign_coeff=0.0, frobenius_coeff=0.0)

Updates the dictionary matrix `D` using the provided learning rate and regularization parameters.

# Arguments
- `D: Dictionary matrix to be updated.
- `lr_D`: Learning rate for updating `D`.
- `X`: Latent state history of the system.
- `Y`: Observations from the system.
- `sign_coeff`: Regularization parameter for sign constraint.
- `frobenius_coeff`: Regularization parameter for Frobenius norm.

# Returns
- Updated dictionary matrix `D`.
"""
function update_D!(
    D::AbstractMatrix{T},
    lr_D::T,
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T};
    sign_coeff::T = zero(T),
    frobenius_coeff::T = zero(T),
) where {T<:AbstractFloat}
    if sign_coeff != 0 || frobenius_coeff != 0
        tmp = similar(Y)
        reconstruction_grad = similar(D)
        mul!(tmp, D, X)                                     # DX
        tmp .= Y .- tmp                                     # Y - DX
        mul!(reconstruction_grad, tmp, X', true, false)     # (Y - DX)X'

        sign_penalty = sum(sign.(D))
        @. D -=
            lr_D * (
                (-2 * reconstruction_grad) +
                (sign_coeff * sign_penalty) +
                (2 * frobenius_coeff * D)
            )
    else
        D .= Y * pinv(X)
    end
end

"""
    step_dynamics(x_t, c_t, F, x_hat_next)

Estimates the next state of the system using the current latent state `x_t` and 
coefficients `c_t`.

# Arguments
- `x_t`: Current latent state of the system.
- `c_t`: Current coefficient state of the system.
- `F`: Dynamics dictionary.
- `x_hat_next`: Destination matrix for computed next state.

# Returns
- Updated state of the system at time `t+1`.
"""
function step_dynamics!(
    x_hat_next::AbstractVector{T},
    x_t::AbstractVector{T},
    c_t::AbstractVector{T},
    F::AbstractArray{T,3},
) where {T<:AbstractFloat}
    fill!(x_hat_next, zero(T))

    for i in axes(F, 1)
        mul!(x_hat_next, @view(F[i, :, :]), x_t, c_t[i], true)
    end
end

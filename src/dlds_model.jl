include("./matrix_utils.jl")

using LinearAlgebra
using Distributions
using Random
using Statistics
using ProximalOperators
using ProximalAlgorithms
using SparseArrays
using Printf
using Lasso
using .MatrixUtils

function train_dLDS()
    """
    train_dLDS()

    Placeholder function for training a dLDS model.
    """
    println("Training dLDS model...")
end

"""
    update_c!(c, FX_prod, X, F, smooth_coeff, l1_coeff; max_iter tol, warm_start)

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
    update_F!(F, gradient_sum, temp_gradient, x_hat_next, residuals, X, c, lr_F, normalize_gradient, normalize_F)

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

"""
    update_X(D, Y, lambda_l1=0.0)

Infers the latent state vector with Lasso regression given the loading matrix D and observations y.

# Arguments
- `D`: Loading matrix.
- `Y`: Observations from the system.
- `lambda_l1`: L1 regularization parameter.

# Returns
- Inferred state vector.
"""
function update_X(
    D::AbstractMatrix{T},
    Y::AbstractMatrix{T};
    lambda_l1::T = zero(T),
) where {T<:AbstractFloat}
    if iszero(lambda_l1)
        return D \ Y
    else
        model = fit(LassoModel, D, Y, λ = lambda_l1)
        return coef(model)
    end
end

"""
    update_D(D, lr_D, x, y; reg_sign=0.0, reg_frobenius=0.0)

Updates the dictionary matrix `D` using the provided learning rate and regularization parameters.

# Arguments
- `D: Dictionary matrix to be updated.
- `lr_D`: Learning rate for updating `D`.
- `X`: Latent state history of the system.
- `Y`: Observations from the system.
- `reg_sign`: Regularization parameter for sign constraint.
- `reg_frobenius`: Regularization parameter for Frobenius norm.

# Returns
- Updated dictionary matrix `D`.
"""
function update_D!(
    D::AbstractMatrix{T},
    lr_D::T,
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T};
    reg_sign::T = zero(T),
    reg_frobenius::T = zero(T),
) where {T<:AbstractFloat}
    if reg_sign != 0 || reg_frobenius != 0
        tmp = similar(Y)
        reconstruction_grad = similar(D)
        mul!(tmp, D, X)                                     # DX
        tmp .= Y .- tmp                                     # Y - DX
        mul!(reconstruction_grad, tmp, X', true, false)     # (Y - DX)X'

        @. D -=
            lr_D * (
                (-2 * reconstruction_grad) +
                (reg_sign * sign(D)) +
                (2 * reg_frobenius * D)
            )
    else
        D .= Y / X
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

# """
#     step_dynamics_multiple(X, coefficients, F, t)å

# Given multiple state vectors representing multiple time points, computes the next
# state prediction for each time point using the provided coefficients.

# # Arguments
# - `X::AbstractMatrix`: Latent state history.
# - `coefficients::AbstractMatrix`: Coefficient history of the system.
# - `F::Vector{<:AbstractMatrix}`: Dynamics dictionary.

# # Returns
# - Updated state of the system at time `t+1`.
# """
# function step_dynamics_multiple(
#     X::AbstractMatrix,
#     coefficients::AbstractMatrix,
#     F::Vector{<:AbstractMatrix};
# )
#     X_next = zeros(size(X, 1), size(X, 2) + 1)
#     for t in 1:size(X, 2)
#         X_next[:, t+1] = step_dynamics(X, coefficients, F, t)
#     end
#     X_next[:, 1] = X[:, 1]
# end
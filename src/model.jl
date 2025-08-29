# NOTE: this function is parallelizable if we do Jacobi updates instead of Gauss-Siedel
# NOTE: this can be made faster with a scheme that uses a loose upper bound Lipschitz estimate instead of doing the search
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
    max_iter::Int = 3000,
    tol::T = 1e-6,
    warm_start::Bool = false,
) where {T<:AbstractFloat}
    L::T = T(0)
    for t in axes(c, 2)
        for i in axes(F, 1)
            mul!(@view(FX_prod[:, i]), @view(F[i, :, :]), @view(X[:, t]))
        end

        if smooth_coeff > 0 && t > 1
            dual_lsq = DoubleLeastSquares(
                FX_prod,
                @view(X[:, t+1]),
                @view(c[:, t-1]),
                2 * smooth_coeff,
            )
        else
            dual_lsq =
                DoubleLeastSquares(FX_prod, @view(X[:, t+1]), nothing, smooth_coeff)
        end

        l1_penalty = NormL1(l1_coeff)

        solver = ProximalAlgorithms.FastForwardBackward(maxit = max_iter, tol = tol)
        initial_guess = warm_start ? @view(c[:, t]) : zeros(T, size(c, 1))

        solution, iters = solver(x0 = initial_guess, f = dual_lsq, g = l1_penalty)
        @view(c[:, t]) .= solution
    end
end

struct DoubleLeastSquares{T<:AbstractFloat}
    FX_prod::AbstractMatrix{T}
    X_tplus1::AbstractVector{T}
    c_tminus1::Union{AbstractVector{T},Nothing}
    lambda::T
end

function ProximalAlgorithms.value_and_gradient(
    double_lsq::DoubleLeastSquares{T},
    c::AbstractVector{T},
) where {T<:AbstractFloat}
    recon_residual = double_lsq.FX_prod * c
    recon_residual .-= double_lsq.X_tplus1
    recon_loss = 0.5 * dot(recon_residual, recon_residual)
    recon_gradient = double_lsq.FX_prod' * recon_residual

    if double_lsq.c_tminus1 !== nothing
        smooth_residual = c - double_lsq.c_tminus1
        smooth_loss = double_lsq.lambda * 0.5 * dot(smooth_residual, smooth_residual)
        axpy!(double_lsq.lambda, smooth_residual, recon_gradient)  # single step to calculate and accumulate gradient

        return recon_loss + smooth_loss, recon_gradient
    else
        return recon_loss, recon_gradient
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

    # NOTE: can potentially batch/multithread this to make it faster
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

        @. @view(F[i, :, :]) += lr_F * @view(gradient_sum[i, :, :])
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
        X .= pinv(D) * Y
    else
        fit = glmnet(
            D,
            Y,
            MvNormal(),
            alpha = 1.0,
            lambda = [lambda_l1],
            intercept = false,
        )
        X .= fit.betas[:, :, 1]
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
        mul!(tmp, D, X)
        tmp .= Y .- tmp
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

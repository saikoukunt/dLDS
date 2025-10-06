function update_c_parallel!(
    c::Vector{<:AbstractMatrix{T}},
    X_trial::Vector{<:AbstractMatrix{T}},
    F::AbstractArray{T,3};
    smooth_coeff::T = zero(T),
    l1_coeff::T = zero(T),
    max_iter::Int = 3000,
    tol::T = 1e-8,
) where {T<:AbstractFloat}
    trial_data = [
        (
            X_trial[i],
            F,
            (
                smooth_coeff = smooth_coeff,
                l1_coeff = l1_coeff,
                max_iter = max_iter,
                tol = tol,
            ),
        ) for i in axes(X_trial, 1)
    ]

    results = pmap(worker_update_c, trial_data)

    for i in axes(c, 1)
        c[i] = results[i]
        if any(isnan.(c[i]))
            println("trial $(i)")
        end
    end
end

function worker_update_c(
    trial_data::Tuple{<:AbstractMatrix{T},<:AbstractArray{T,3},NamedTuple},
) where {T<:AbstractFloat}
    X, F, kwargs = trial_data

    num_latents = size(X, 1)
    num_motifs = size(F, 1)

    FX_prod = Matrix{T}(undef, num_latents, num_motifs)
    FX_prod_gram = Matrix{T}(undef, num_motifs, num_motifs)

    return update_c!(
        zeros(size(F, 1), size(X, 2) - 1),
        FX_prod,
        FX_prod_gram,
        X,
        F;
        kwargs...,
    )
end

# NOTE: this function is parallelizable across time if we do Jacobi updates instead of Gauss-Siedel
"""
    update_c!(c, FX_prod, FX_prod_gram, X, F, smooth_coeff, l1_coeff; max_iter, tol, warm_start)

Calculates an update estimate of dynamics motif coefficients.

# Arguments

- `c`: Dynamics coefficient history of the system.
- `FX_prod`: A preallocated (# of latents X # of motifs) matrix to hold intermediate results.
- `FX_prod_gram`: A preallocated (# of motifs X # of motifs) matrix to hold intermediate results
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
    FX_prod_gram::AbstractMatrix{T},
    X::AbstractMatrix{T},
    F::AbstractArray{T,3};
    smooth_coeff::T = zero(T),
    l1_coeff::T = zero(T),
    max_iter::Int = 3000,
    tol::T = 1e-8,
) where {T<:AbstractFloat}
    L::T = T(0)
    lambda_l1 = Vector{T}(undef, size(c, 1))
    lambda_l1 .= l1_coeff
    l1_penalty = NormL1(lambda_l1)
    zero_guess = zeros(T, size(c, 1))
    solver = ProximalAlgorithms.FastForwardBackward(maxit = max_iter, tol = tol)
    double_lsq =
        DoubleLeastSquares(FX_prod, @view(X[:, 1]), @view(c[:, 1]), smooth_coeff)

    for t in axes(c, 2)
        for i in axes(F, 1)
            mul!(@view(FX_prod[:, i]), @view(F[i, :, :]), @view(X[:, t]))
        end

        # TODO: profile precomputing L vs letting the solver do a backtracking line search
        LinearAlgebra.BLAS.syrk!('U', 'T', true, FX_prod, false, FX_prod_gram)
        L = eigmax(Symmetric(FX_prod_gram, :U))
        if smooth_coeff > 0 && t > 1
            update_double_least_squares!(double_lsq, @view(X[:, t+1]), @view(c[:, t-1]))
            L += smooth_coeff
        else
            update_double_least_squares!(double_lsq, @view(X[:, t+1]), Inf)
        end

        solution, iters =
            solver(x0 = zero_guess, f = double_lsq, g = l1_penalty, Lf = L)
        if any(isnan.(solution))
            c[:, t] = zeros(size(solution))
            continue
        end

        @. lambda_l1 = l1_coeff / (1 + 200 * (abs(solution))) # reweight L1 to correct for bias
        solution, iters = solver(x0 = solution, f = double_lsq, g = l1_penalty, Lf = L)

        c[:, t] = solution
    end

    return c
end

mutable struct DoubleLeastSquares{T<:AbstractFloat}
    FX_prod::AbstractMatrix{T}
    X_tplus1::AbstractVector{T}
    c_tminus1::Union{AbstractVector{T},T}
    lambda::T
end

function update_double_least_squares!(
    double_lsq::DoubleLeastSquares,
    X_tplus1::AbstractVector{T},
    c_tminus1::Union{AbstractVector{T},T},
) where {T<:AbstractFloat}
    double_lsq.X_tplus1 = X_tplus1
    double_lsq.c_tminus1 = c_tminus1
end

function ProximalAlgorithms.value_and_gradient(
    double_lsq::DoubleLeastSquares{T},
    c::AbstractVector{T},
) where {T<:AbstractFloat}
    recon_residual = double_lsq.FX_prod * c
    recon_residual .-= double_lsq.X_tplus1
    recon_loss = 0.5 * dot(recon_residual, recon_residual)
    recon_gradient = double_lsq.FX_prod' * recon_residual

    if double_lsq.c_tminus1 isa AbstractVector{T}
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
    X::Vector{<:AbstractMatrix{T}},
    c::Vector{<:AbstractMatrix{T}},
    lr_F::T;
    decorr_coeff::T = T(0.05),
    normalize_F::Bool = true,
) where {T<:AbstractFloat}
    fill!(gradient_sum, zero(T))
    num_timepoints = size(X[1], 2) - 1

    # Calculate the sum of the latent reconstruction gradients over time w.r.t each F
    for trial in axes(c, 1)
        for t in 1:num_timepoints
            step_dynamics!(x_hat_next, @view(X[trial][:, t]), @view(c[trial][:, t]), F)
            residuals .= @view(X[trial][:, t+1]) .- x_hat_next
            mul!(temp_gradient, residuals, @view(X[trial][:, t])')

            for i in axes(F, 1)
                axpy!(c[trial][i, t], temp_gradient, @view(gradient_sum[i, :, :]))
            end
        end
    end

    # Take the step
    F .+= lr_F / (num_timepoints * size(c, 1)) .* gradient_sum

    # Decorrelate
    gradient_sum .= 0
    for i in axes(F, 1)
        for j in axes(F, 1)
            if i != j
                # compute the trace without a new implicit allocation
                trace = T(0)
                for k in axes(F, 3)
                    trace += dot(@view(F[i, :, k]), @view(F[j, :, k]))
                end

                # add the decorrelation term to the running gradient
                axpy!(trace, F[j, :, :], @view(gradient_sum[i, :, :]))
            end
        end
    end
    @. F -= decorr_coeff * gradient_sum

    for i in axes(F, 1)
        if normalize_F
            normalize_matrix!(@view(F[i, :, :]))
        end
    end
end

# TODO: Fuse this with update_c via stacking
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

        sign_penalty = mapreduce(sign, +, D)
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
    step_dynamics(x_hat_next, x_t, c_t, F)

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

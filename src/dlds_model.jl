using LinearAlgebra
using Distributions
using Random
using Statistics
using ProximalOperators
using ProximalAlgorithms
using SparseArrays
using Printf
using Lasso

function train_dLDS()
    """
    train_dLDS()

    Placeholder function for training a dLDS model.
    """
    println("Training dLDS model...")
end

function update_c(
    X::AbstractMatrix,
    F::Vector{<:AbstractMatrix},
    coefficients::AbstractMatrix,
    params_update_c::Dict{String,Any} = Dict(
        "update_c_type" => "inv",
        "reg_term" => 0,
        "smooth_term" => 0,
        "to_norm_fx" => false,
    ),
    other_params::Dict{String,Any},
    random_state::Int = 0,
    direction::String = "c2n",
    skip_error::Bool = false,
    X_clear::AbstractMatrix = nothing,
) end

function update_f_all(
    X::AbstractMatrix,
    F::Vector{<:AbstractMatrix},
    coefficients::AbstractMatrix,
    lr_F::Float64,
    normalize::Bool = false,
    reduction::String = "mean",
    normalize_eig::Bool = true,
)
    if reduction != "mean" && reduction != "sum"
        error("reduction must be either 'mean' or 'sum'")
    end

    F_new = Vector{<:AbstractMatrix}(undef, length(F))
    gradients = calculate_ci_fi_xt(X, F, coefficients)
    for i in 1:length(F)
        if reduction == "mean"
            gradient_dir =
                mean(gradients[:, :, :] * reshape(coefficients[i], (1, 1, :)), dims = 3)
        elseif reduction == "median"
            gradient_dir = median(
                gradients[:, :, :] * reshape(coefficients[i], (1, 1, :)),
                dims = 3,
            )
        end

        if normalize
            gradient_dir = normalize_matrix!(gradient_dir, "eigen")
        end
        F_new[i] = F[i] - 2 * lr_F * gradient_dir
        if normalize_eig
            F_new[i] = normalize_matrix!(F_new[i], "eigen")
        end
    end

    # replace NaN entries with random values uniform from 0 to 1
    for i in 1:length(F)
        F_new[i][isnan.(F_new[i])] .= rand(Uniform(0, 1), sum(isnan.(F_new[i])))
    end

    return F_new
end

function calculate_ci_fi_xt(
    X::AbstractMatrix,
    F::Vector{<:AbstractMatrix},
    coefficients::AbstractMatrix,
    cumulative::Bool = false,
)
    gradients = zeros(size(X, 1), size(X, 1), size(X, 2) - 1)

    local x_hat_next
    for t in 1:axes(X, 2)-1
        if cumulative
            x_prev = t > 1 ? x_hat_next : X[:, 1]
            x_hat_next = step_dynamics(x_prev, coefficients, F, t)
            gradients[:, :, t] = (X[:, t+1] - x_hat_next) * transpose(x_prev)
        else
            x_hat_next = step_dynamics(X[:, t], coefficients, F, t)
            gradients[:, :, t] = (X[:, t+1] - x_hat_next) * transpose(X[:, t])
        end
    end

    return gradients
end

function update_X(
    D::AbstractMatrix,
    y::AbstractMatrix,
    lambda_l1::Float64 = 0.0,
    random_state::Int = 0,
    other_params::Dict{String,Any} = Dict(),
)
    """
    update_X(D, y, lambda_l1=0.0, random_state=0, other_params)

    Infers the latent state vector with Lasso regression given the loading matrix D and observations y.

    # Arguments
    - `D::AbstractMatrix`: Loading matrix.
    - `y::AbstractMatrix`: Observation matrix.
    - `lambda_l1::Float64`: L1 regularization parameter.
    - `random_state::Int`: Seed for solver.
    - `other_params::Dict{String, Any}`: Additional parameters for the update.

    # Returns
    - Inferred state vector.
    """
    if lambda_l1 == 0.0
        return y * pinv(D)
    else
        model = fit(LassoModel, D, y, λ = lambda_l1)
        return coef(model)
    end
end

function update_D(
    D::AbstractMatrix,
    lr_D::Float64,
    x::AbstractMatrix,
    y::AbstractMatrix,
    reg_sign::Float64 = 0.0,
    reg_frobenius::Float64 = 0.0,
)
    """
    update_D(D, lr_D, x, y, reg_sign=0.0, reg_frobenius=0.0)

    Updates the dictionary matrix `D` using the provided learning rate and regularization parameters.

    # Arguments
    - `D::AbstractMatrix`: Dictionary matrix to be updated.
    - `lr_D::Float64`: Learning rate for updating `D`.
    - `x::AbstractMatrix`: Input data matrix.
    - `y::AbstractMatrix`: Target data matrix.
    - `reg_sign::Float64`: Regularization parameter for sign constraint.
    - `reg_frobenius::Float64`: Regularization parameter for Frobenius norm.

    # Returns
    - Updated dictionary matrix `D`.
    """
    if reg_sign == 0 && reg_frobenius == 0
        D_new = y * pinv(x)
    else
        reconstruction_grad = -2 * (y - D * x) * transpose(x)
        sign_grad = reg_sign == 0 ? zeros(size(D)) : reg_sign * sign.(D) # i dont know what this is for
        frobenius_grad = reg_frobenius == 0 ? zeros(size(D)) : 2 * reg_frobenius * D

        D_new = D - lr_D * (reconstruction_grad + sign_grad + frobenius_grad)
    end

    return D_new
end

function step_dynamics_multiple(
    X::AbstractMatrix,
    coefficients::AbstractMatrix,
    F::Vector{<:AbstractMatrix},
    smooth_coeffs::Bool = false,
    smoothing_params = {"window" => 5},
)
    """
    step_dynamics_multiple(X, coefficients, F, t)

    Given multiple state vectors representing multiple time points, computes the next
    state prediction for each time point using the provided coefficients.

    # Arguments
    - `X::AbstractMatrix`: State history of the system.
    - `coefficients::AbstractMatrix`: Coefficient history of the system.
    - `F::Vector{<:AbstractMatrix}`: Dynamics dictionary.

    # Returns
    - Updated state of the system at time `t+1`.
    """
    if smooth_coeffs
        coefficients = smooth_coefficients(coefficients, smoothing_params) #TODO: Implement smoothing
    end

    X_next = zeros(size(X, 1), size(X, 2) + 1)
    for t in 1:size(X, 2)
        X_next[:, t+1] = step_dynamics(X, coefficients, F, t)
    end
    X_next[:, 1] = X[:, 1]
end

function step_dynamics(
    X::AbstractMatrix,
    coefficients::AbstractMatrix,
    F::Vector{<:AbstractMatrix},
    t::Int,
)
    """
    step_dynamics(X, coefficients, F, t)

    Computes the next state of the system at time `t` using the current state `X` and the coefficients.

    # Arguments
    - `X::AbstractMatrix`: Current state or state history of the system.
    - `coefficients::AbstractMatrix`: Coefficient history of the system.
    - `F::Vector{<:AbstractMatrix}`: Dynamics dictionary.
    - `t::Int`: Current time step.

    # Returns
    - Updated state of the system at time `t+1`.
    """

    x_t = size(X, 2) > 1 ? X[:, t] : X
    x_next = zeros(size(x_t))
    for i in 1:length(F)
        mul!(x_next, F[i], x_t, coefficients[i, t], 1.0) # inplace multiply-add to reduce allocation overhead
    end

    return x_next
end

function init_matrix(
    matrix_size::Tuple{Int,Int},
    random_seed::Int,
    distribution::String = "norm",
    init_params::Dict{String,Float64} = Dict("loc" => 0.0, "scale" => 1.0),
    normalize::Bool = false,
)
    """
        init_matrix(matrix_size, random_seed, distribution="normal", init_params, normalize)

    Initializes a matrix of given size with specified distribution and parameters.

    # Arguments
    - `matrix_size::Tuple{Int,Int}`: Size of the matrix to initialize.
    - `random_seed::Int`: Seed for random number generation.
    - `distribution::String`: Type of distribution to use for initialization. Options are:
        - `"normal"`: Normal distribution.
        - `"uniform"`: Uniform distribution.
        - `"integers"`: Constant value.
        - `"sparse"`: Sparse matrix with random non-zero entries.
        - '"regional"`: Regional initialization with a specific pattern.`
    - `init_params::Dict{String, Float64}`: Parameters for the distribution.
        - For `"normal"`: `loc` (mean), `scale` (standard deviation).
        - For `"uniform"`: `low`, `high`.
        - For `"integers"`: `low`, `high`.
        - For `"sparse"`: `k` (max number of non-zero entries).
        - For `"regional"`: `k` (number of subdynamics repeats)
    - `normalize::Bool`: Whether to normalize the matrix after initialization.
    """

    Random.seed!(random_seed)

    if distribution == "normal"
        loc = get(init_params, "loc", 0.0)
        scale = get(init_params, "scale", 1.0)
        random_mat = rand(Normal(loc, scale), matrix_size)
    elseif distribution == "uniform"
        low = get(init_params, "low", 0.0)
        high = get(init_params, "high", 1.0)
        random_mat = rand(Uniform(low, high), matrix_size)
    elseif distribution == "integers"
        low = get(init_params, "low", 0)
        high = get(init_params, "high", 10)
        random_mat = rand(low:high, matrix_size)
    elseif distribution == "sparse"
        k = get(init_params, "k", 1)  # Number of non-zero entries
        rows, cols = matrix_size
        random_mat = zeros(rows, cols)
        for col in 1:cols
            num_nonzero = rand(1:min(k, rows))
            rows_indices = rand(1:rows, num_nonzero)
            random_mat[rows_indices, col] = 1
        end
    elseif distribution == "regional"
        # leaving this as a placeholder for future implementation
    else
        error(
            "Unknown distribution type: $distribution. Use 'normal', 'uniform', 'integers', or 'sparse'.",
        )
    end

    if normalize
        return normalize_matrix!(random_mat)
    else
        return random_mat
    end
end

function normalize_matrix!(A::AbstractMatrix, norm_type::String = "eigen")
    """
        normalize_matrix!(A, norm_type="eigen")

    Applies the specified normalization to the matrix `A` in-place.

    # Arguments
    - `A::AbstractMatrix`: The matrix to normalize.
    - `norm_type::String`: The type of normalization to apply. Options are:
        - `"eigen"`: Normalize by the largest eigenvalue.
        - `"max"`: Normalize by the maximum absolute value.
        - `"exp"`: Normalize using the matrix exponential.
    """

    if norm_type == "eigen"
        return A / maximum(abs.(eigvals(A)))
    elseif norm_type == "max"
        return A / maximum(abs.(A))
    elseif norm_type == "exp"
        return exp(-tr(mat)) * exp(mat)
    else
        error("Unknown normalization type: $norm_type. Use 'eigen', 'max', or 'exp'.")
    end
end

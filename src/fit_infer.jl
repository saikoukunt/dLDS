# TODO: add F perturbing and initial correlation check
function fit_full_model(
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
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 3000,
    D_lr::T = T(30),
    D_sign_coeff::T = zero(T),
    D_frobenius_coeff::T = zero(T),
    F_lr_init::T = T(30),
    F_normalize_matrix::Bool = true,
    F_normalize_gradient::Bool = false,
    F_perturb_threshold::T = T(1e-5),
    F_noise_sigma::T = T(0.1),
    F_init_max_corr::T = zero(T),
    F_lr_decay::T = T(0.99995),
    verbose::Bool = true,
) where {T<:AbstractFloat}
    """
    train_dLDS()

    """
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
    #validate_F_separation!(F_init_max_corr) #TODO: Implement this
    c::Matrix{T} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_timepoints - 1),
        random_seed,
    )
    X::Matrix{T} = Matrix{T}(undef, num_latents, num_timepoints)

    F_lr::T = F_lr_init
    i::Int = 1
    latent_recon_err::Vector{T} = zeros(T, num_timepoints)
    data_recon_err = Inf

    # Pre-allocate for intermediate results
    data_prediction::Matrix{T} = similar(Y)
    FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs) # for update_c!
    FX_prod_gram::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
    gradient_sum::Array{T,3} = similar(F)                          # for update f! 
    temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
    x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
    update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

    while (data_recon_err > recon_threshold) && (i <= max_iter)
        update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

        c_l1_coeff *= c_l1_coeff_decay
        if i > 1
            update_c!(
                c,
                FX_prod,
                FX_prod_gram,
                X,
                F,
                smooth_coeff = c_smooth_coeff,
                l1_coeff = c_l1_coeff;
                tol = c_fista_tol,
                max_iter = c_fista_max_iter,
            )
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

function fit_no_obs_model(
    X::Vector{<:AbstractMatrix{T}},
    num_motifs::Int;
    samples_per_snippet::Int = 200,
    num_snippets::Int = 50,
    random_seed::Int = 0,
    max_iter::Int = 5000,
    recon_threshold::T = T(1e-5),
    c_l1_coeff::T = T(0.2),
    c_l1_coeff_decay::T = T(1),
    c_smooth_coeff::T = T(0.4),
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 1000,
    F_lr_init::T = T(1),
    F_normalize_matrix::Bool = true,
    F_decorr_coeff::T = T(0.0),
    F_perturb_threshold::T = T(1e-5),
    F_noise_sigma::T = T(0.1),
    F_init_max_corr::T = zero(T),
    F_lr_decay::T = T(0.99995),
    verbose::Bool = true,
) where {T<:AbstractFloat}
    num_latents::Int = size(X[1], 1)
    num_timepoints::Int = size(X[1], 2)

    # Initialize model parameters and state
    F::Array{T,3} =
        # create_random_dynamics(num_latents, num_motifs, correlation_tol = 0.1)

    init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_latents, num_latents),
        random_seed,
    )
    #validate_F_separation!(F_init_max_corr) #TODO: Implement this

    c = Vector{Matrix{T}}(undef, num_snippets)
    for i in 1:num_snippets
        c[i] = zeros(num_motifs, samples_per_snippet - 1)
    end

    F_lr::T = F_lr_init
    i::Int = 1
    latent_recon_err = Inf

    gradient_sum::Array{T,3} = similar(F)                          # for update f! 
    temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
    x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
    update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

    while (i <= max_iter)
        F_old = copy(F)
        X_snippets, _ = sample_snippets(X, num_snippets, samples_per_snippet)

        update_c_parallel!(
            c,
            X_snippets,
            F;
            smooth_coeff = c_smooth_coeff,
            l1_coeff = c_l1_coeff,
            max_iter = c_fista_max_iter,
            tol = c_fista_tol,
        )

        update_F!(
            F,
            gradient_sum,
            temp_gradient,
            x_hat_next,
            update_F_residuals,
            X_snippets,
            c,
            F_lr;
            normalize_F = F_normalize_matrix,
            decorr_coeff = F_decorr_coeff,
        )
        c_l1_coeff *= c_l1_coeff_decay
        F_lr *= F_lr_decay

        if verbose
            latent_recon_err =
                calculate_latent_recon_error!(x_hat_next, F, X_snippets, c)
            dF = calculate_delta_F(F, F_old)
            println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")
        end
        i += 1
    end

    return F
end

function infer_no_obs_state(
    F::AbstractArray{T,3},
    X::AbstractMatrix{T};
    c_l1_coeff::T = zero(T),
    c_smooth_coeff::T = zero(T),
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 1000,
    random_seed::Int = 0,
) where {T<:AbstractFloat}
    num_motifs = size(F, 1)
    num_timepoints = size(X, 2)
    num_latents = size(X, 1)

    c::Matrix{T} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_timepoints - 1),
        random_seed,
    )

    FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs)
    FX_prod_gram::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
    update_c!(
        c,
        FX_prod,
        FX_prod_gram,
        X,
        F,
        smooth_coeff = c_smooth_coeff,
        l1_coeff = c_l1_coeff;
        tol = c_fista_tol,
        max_iter = c_fista_max_iter,
    )

    return c
end

function infer_full_state(
    D::AbstractMatrix{T},
    F::AbstractArray{T,3},
    Y::AbstractMatrix{T};
    x_l1_coeff::T = zero(T),
    c_l1_coeff::T = zero(T),
    c_smooth_coeff::T = zero(T),
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 1000,
) where {T<:AbstractFloat}
    X::Matrix{T} = Matrix{T}(undef, num_latents, num_timepoints)
    update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

    c = infer_no_obs_state(
        F,
        X;
        c_l1_coeff = c_l1_coeff,
        c_smooth_coeff = c_smooth_coeff,
        c_fista_tol = c_fista_tol,
        c_fista_max_iter = c_fista_max_iter,
    )

    return X, c
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
    X::Vector{<:AbstractMatrix{T}},
    c::Vector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    total_error::T = zero(T)

    for trial in axes(c, 1)
        for t in 1:size(X[trial], 2)-1
            step_dynamics!(x_hat_next, @view(X[trial][:, t]), @view(c[trial][:, t]), F)
            x_hat_next .= @view(X[trial][:, t+1]) .- x_hat_next    # residual, reusing array to avoid extra allocation

            total_error += dot(x_hat_next, x_hat_next)
        end
    end

    return total_error / sum([sum(X[trial] .^ 2) for trial in axes(c, 1)])
end

function calculate_delta_F(
    F_new::AbstractArray{T,3},
    F_old::AbstractArray{T,3},
) where {T<:AbstractFloat}
    delta = 0
    num_motifs = size(F_old, 1)

    for i in 1:num_motifs
        delta +=
            1 / num_motifs *
            sum((@view(F_old[i, :, :]) .- @view(F_new[i, :, :])) .^ 2) /
            sum(@view(F_new[i, :, :]) .^ 2)
    end

    return delta
end
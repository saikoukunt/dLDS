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
    c_fista_max_iter::Int = 1000,
    D_lr::T = T(30),
    D_sign_coeff::T = zero(T),
    D_frobenius_coeff::T = zero(T),
    F_lr_init::T = T(30),
    F_normalize_matrix::Bool = true,
    F_normalize_gradient::Bool = false,
    F_perturb_threshold::T = T(1e-5),
    F_noise_sigma::T = T(0.1),
    F_init_max_corr::T = zero(T),
    F_lr_decay::T = T(0.999),
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
                X,
                F,
                smooth_coeff = c_smooth_coeff,
                l1_coeff = c_l1_coeff;
                tol = c_fista_tol,
                max_iter = c_fista_max_iter,
                warm_start = true,
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
    X::AbstractMatrix{T},
    num_motif::Int;
    random_seed::Int = 0,
    max_iter::Int = 3000,
    c_l1_coeff::T = zero(T),
    c_l1_coeff_decay::T = T(1),
    c_smooth_coeff::T = zero(T),
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 1000,
    F_normalize_matrix::Bool = true,
    F_normalize_gradient::Bool = false,
    F_perturb_threshold::T = T(1e-5),
    F_noise_sigma::T = T(0.1),
    F_init_max_corr::T = zero(T),
    F_lr_decay::T = T(0.999),
    verbose::Bool = true,
) where {T<:AbstractFloat}
    num_latents::Int = size(X, 1)
    num_timepoints::Int = size(X, 2)

    # Initialize model parameters and state
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

    F_lr::T = F_lr_init
    i::Int = 1
    latent_recon_err::Vector{T} = zeros(T, num_timepoints)

    FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs) # for update_c!
    gradient_sum::Array{T,3} = similar(F)                          # for update f! 
    temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
    x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
    update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)

    while (latent_recon_err > recon_threshold) && (i <= max_iter)
        update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

        c_l1_coeff *= c_l1_coeff_decay
        update_c!(
            c,
            FX_prod,
            X,
            F,
            smooth_coeff = c_smooth_coeff,
            l1_coeff = c_l1_coeff;
            tol = c_fista_tol,
            max_iter = c_fista_max_iter,
            warm_start = i > 1 ? c_fista_warm_start : false,
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

        latent_recon_err[i] = calculate_latent_recon_error!(x_hat_next, F, X, c)

        if verbose
            println(
                "Iter $(i): Data Rec. Error: $(data_recon_err), Latent Rec. Error: $(latent_recon_err[i]) ",
            )
        end
        i += 1
    end

    return F, c, latent_recon_err
end

function infer_no_obs_state(
    F::AbstractArray{T,3},
    X::AbstractMatrix{T};
    c_l1_coeff::T = zero(T),
    c_smooth_coeff::T = zero(T),
    c_fista_tol::T = T(1e-8),
    c_fista_max_iter::Int = 1000,
) where {T<:AbstractFloat}
    c::Matrix{T} = init_matrix(
        InitDistribution.Normal(),
        (num_motifs, num_timepoints - 1),
        random_seed,
    )

    FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs)
    update_c!(
        c,
        FX_prod,
        X,
        F,
        smooth_coeff = c_smooth_coeff,
        l1_coeff = c_l1_coeff;
        tol = c_fista_tol,
        max_iter = c_fista_max_iter,
        warm_start = i > 1 ? c_fista_warm_start : false,
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
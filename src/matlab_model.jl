
function matlab_update_F!(
    F::AbstractArray{T,3},
    X::AbstractMatrix{T},
    c::AbstractMatrix{T};
    lr_F::T = T(0.01),
    lambda_F::T = T(0),
) where {T<:AbstractFloat}
    X_tplus1 = @view(X[:, 2:end])
    X_t = @view(X[:, 1:end-1])
    c_t = @view(c[:, 2:end])

    F_grad = zeros(size(F))
    residuals = zeros(size(X_t))
    for t in axes(X_t, 2)
        F_tmp = dyn_cell2mat(F, c_t[:, t])
        residuals[:, t] .= X_tplus1[:, t] - F_tmp * X_t[:, t]
    end

    for t in axes(residuals, 2)
        tmp_mat = residuals[:, t] * X_t[:, t]'
        for i in axes(F, 1)
            F_grad[i, :, :] .+= c_t[i, t] * tmp_mat
        end
    end

    F_new = similar(F)
    for i in axes(F, 1)
        F_new[i, :, :] = F[i, :, :] + (lr_F / size(residuals, 2)) * F_grad[i, :, :]
        F_new[i, :, :] = project_down!(F_new[i, :, :], lambda_F, F, i)
    end
    F .= F_new

    return F
end

function dyn_cell2mat(
    F::AbstractArray{T,3},
    c::AbstractVector{T},
) where {T<:AbstractFloat}
    F_mat = zeros(size(F, 2), size(F, 3))
    for i in axes(F, 1)
        F_mat .+= F[i, :, :] * c[i]
    end
    return F_mat
end

function project_down!(
    F_new::AbstractMatrix{T},
    lambda_F::T,
    F::AbstractArray{T,3},
    i::Int,
) where {T<:AbstractFloat}
    for j in axes(F, 1)
        if i !== j
            F_new = F_new - lambda_F * tr(F[i, :, :]' * F[j, :, :]) * F[j, :, :]
        end
    end

    max_svd = maximum(abs.(svd(F_new).S))
    if iszero(max_svd)
        F_new .= F[i, :, :]
    else
        F_new ./= max_svd
    end

    return F_new
end
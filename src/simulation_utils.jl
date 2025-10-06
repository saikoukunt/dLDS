function create_random_dynamics(dimension::Int, num_motifs::Int; correlation_tol = 0.02)
    F = zeros(num_motifs, dimension, dimension)
    for i in axes(F, 1)
        repeat = true
        while repeat
            repeat = false
            F[i, :, :] = sample_orthogonal_matrix(dimension)

            for j in 1:i-1
                if abs(cor(vec(F[i, :, :]), vec(F[j, :, :]))) > correlation_tol
                    repeat = true
                end
            end
        end
    end

    return F
end

"""
    sample_orthogonal_matrix(num_neurons)

Generates a random square dynamics matrix by drawing from the uniform distribution over 
the space of orthogonal matrices.

# Arguments
- `dimension`: Number of neurons/latents.

# Returns
A random orthogonal matrix.
"""
function sample_orthogonal_matrix(dimension::Int)
    Q, R = qr(randn(dimension, dimension))
    F = Q * Diagonal(sign.(diag(R)))

    return F
end

function generate_switching_c!(
    c::AbstractMatrix{T},
    num_timepoints::Int,
    num_motifs::Int;
    min_switch_time::Int = 100,
    max_extra_switch_time::Int = 300,
) where {T<:AbstractFloat}
    fill!(c, T(0))
    t = 1
    while t < num_timepoints
        active_length = min_switch_time + rand(1:max_extra_switch_time)
        end_time = min(t + active_length, num_timepoints)
        active_system = rand(0:num_motifs)

        if active_system !== 0
            c[active_system, t:end_time] .= 1
        end
        t = end_time + 1
    end

    return c
end

function states_to_observations() end
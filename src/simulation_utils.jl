"""
    create_random_dynamics(num_neurons)

Generates a random square dynamics matrix by drawing from the uniform distribution over 
the space of diagonal matrices.

# Arguments
- `dimension`: Number of neurons/latents.

# Returns
A random orthogonal matrix.
"""
function create_random_dynamics(dimension::Int)
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
        active_system = rand(1:num_motifs)

        if active_system !== 0
            c[active_system, t:end_time] .= 1
        end
        t = end_time
    end

    return c
end

function states_to_observations() end
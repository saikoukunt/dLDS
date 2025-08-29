function simulate_two_subsystems_no_obs(
    num_timepoints::Int,
    num_neurons_per_system::AbstractVector{Int},
    num_motifs_per_system::AbstractVector{Int},
    num_trials::Int;
    switching::Bool = true,
    min_switch_time::Int = 50,
    max_extra_switch_time::Int = 150,
)
    num_neurons = sum(num_neurons_per_system)
    num_motifs = sum(num_motifs_per_system)

    F = zeros(Float64, num_motifs, num_neurons, num_neurons)
    for i in 1:num_motifs_per_system[1]
        F[i, 1:num_neurons_per_system[1], 1:num_neurons_per_system[1]] =
            create_random_dynamics(num_neurons_per_system[1])
    end
    for i in num_motifs_per_system[1]+1:num_motifs
        F[i, num_neurons_per_system[1]+1:end, num_neurons_per_system[1]+1:end] =
            create_random_dynamics(num_neurons_per_system[2])
    end

    if switching
        X, c = simulate_two_switching_systems(
            F,
            num_trials,
            num_timepoints,
            num_neurons_per_system,
            num_motifs_per_system;
            min_switch_time,
            max_extra_switch_time,
        )
    else
        print("Not implemented yet!")
    end

    return X, c, F
end

# TODO: fix the Xs going to 0
function simulate_two_switching_systems(
    F::AbstractArray{T,3},
    num_trials::Int,
    num_timepoints::Int,
    num_neurons_per_system::AbstractVector{Int},
    num_motifs_per_system::AbstractVector{Int};
    min_switch_time::Int = 100,
    max_extra_switch_time::Int = 300,
) where {T<:AbstractFloat}
    num_neurons = sum(num_neurons_per_system)
    num_motifs = sum(num_motifs_per_system)

    X = zeros(Float64, num_trials, num_neurons, num_timepoints)
    c = zeros(Float64, num_trials, num_motifs, num_timepoints)

    for trial in 1:num_trials
        generate_switching_c!(
            @view(c[trial, 1:num_motifs_per_system[1], :]),
            num_timepoints,
            num_motifs_per_system[1];
            min_switch_time = min_switch_time,
            max_extra_switch_time = max_extra_switch_time,
        )
        generate_switching_c!(
            @view(c[trial, num_motifs_per_system[1]+1:end, :]),
            num_timepoints,
            num_motifs_per_system[2];
            min_switch_time = min_switch_time,
            max_extra_switch_time = max_extra_switch_time,
        )
        F_t = generate_two_subsystem_state_trajectory!(
            @view(X[trial, :, :]),
            @view(c[trial, :, :]),
            F,
            num_timepoints,
            num_neurons_per_system,
            num_motifs_per_system,
        )
    end

    return X, c
end

function generate_two_subsystem_state_trajectory!(
    X::AbstractMatrix{T},
    c::AbstractMatrix{T},
    F::AbstractArray{T,3},
    num_timepoints::Int,
    num_neurons_per_system::AbstractVector{Int},
    num_motifs_per_system::AbstractVector{Int},
) where {T<:AbstractFloat}
    F_t::Matrix{T} = Matrix{T}(undef, size(F, 2), size(F, 2))
    X[:, 1] = randn(T, size(X, 1))

    for t in 2:num_timepoints
        fill!(F_t, T(0))
        for motif in axes(F, 1)
            axpy!(c[motif, t], F[motif, :, :], F_t)
        end

        if all(X[1:num_neurons_per_system[1], t-1] .== T(0)) &&
           any(c[1:num_motifs_per_system[1], t] .!== T(0))
            X[1:num_neurons_per_system[1], t-1] = randn(T, num_neurons_per_system[1])
        end
        if all(X[num_neurons_per_system[1]+1:end, t-1] .== T(0)) &&
           any(c[num_motifs_per_system[1]+1:end, t] .!== T(0))
            X[num_neurons_per_system[1]+1:end, t-1] =
                randn(T, num_neurons_per_system[2])
        end

        X[:, t] = F_t * @view(X[:, t-1])
    end

    return X
end
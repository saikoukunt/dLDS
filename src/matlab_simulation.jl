function matlab_simulation(
    num_timepoints::Int,
    num_neurons::Vector{Int},
    num_motifs::Vector{Int},
    num_trials::Int,
)
    total_neurons = sum(num_neurons)
    F_circuit_1 = Array{Float64,3}(undef, num_motifs[1], num_neurons[1], num_neurons[1])
    F_circuit_2 = Array{Float64,3}(undef, num_motifs[2], num_neurons[2], num_neurons[2])

    for i in axes(F_circuit_1, 1)
        F_circuit_1[i, :, :] = sample_random_dynamics(num_neurons[1])
    end
    for i in axes(F_circuit_2, 1)
        F_circuit_2[i, :, :] = sample_random_dynamics(num_neurons[2])
    end

    top_rows = Array{Float64,3}(undef, num_motifs[1], num_neurons[1], total_neurons)
    bottom_rows = Array{Float64,3}(undef, num_motifs[2], num_neurons[2], total_neurons)
    for i in axes(F_circuit_1, 1)
        top_rows[i, :, :] =
            cat(F_circuit_1[i, :, :], zeros(num_neurons[1], num_neurons[2]); dims = 2)
    end
    for i in axes(F_circuit_2, 1)
        bottom_rows[i, :, :] =
            cat(zeros(num_neurons[2], num_neurons[1]), F_circuit_2[i, :, :]; dims = 2)
    end

    X, c = generate_random_switched_system(
        num_trials,
        num_neurons,
        num_timepoints,
        top_rows,
        bottom_rows,
    )

    F = zeros(sum(num_motifs), total_neurons, total_neurons)
    F[1:num_motifs[1], 1:num_neurons[1], :] .= top_rows
    F[num_motifs[1]+1:end, num_neurons[1]+1:end, :] .= bottom_rows

    return X, c, F
end

function sample_random_dynamics(num_neurons::Int)
    tol = 1e-5
    F = zeros(num_neurons, num_neurons)

    vi = randn(num_neurons)
    F[:, 1] = vi ./ norm(vi)

    for i in 2:num_neurons
        nrm = 0
        while nrm < tol
            vi = randn(num_neurons)
            vi = vi - F[:, 1:i-1] * (F[:, 1:i-1]' * vi)
            nrm = norm(vi)
        end
        F[:, i] = vi ./ nrm
    end

    return F
end

function generate_random_switched_system(
    num_trials::Int,
    num_neurons::Vector{Int},
    num_timepoints::Int,
    top_rows::AbstractArray{T,3},
    bottom_rows::AbstractArray{T,3},
) where {T<:AbstractFloat}
    total_neurons = sum(num_neurons)
    total_motifs = size(top_rows, 1) + size(bottom_rows, 1)

    c = Vector{Matrix{T}}(undef, num_trials)
    X = Vector{Matrix{T}}(undef, num_trials)

    for trial in axes(c, 1)
        c[trial] = zeros(total_motifs, num_timepoints)
        X[trial] = ones(total_neurons, num_timepoints)
    end

    for trial in 1:num_trials
        X[trial][:, 1] .= randn(total_neurons)
        X[trial][:, 1] ./= norm(X[trial][:, 1])

        t_now = 0
        while t_now < num_timepoints - 1
            t_jump = Int(round(rand()) * 300 + 100)
            t_jump = t_jump - max(t_now + t_jump - num_timepoints + 1, 0)
            F, selF = make_F(top_rows, bottom_rows)

            # re-initalize parts of X_t if necessary and normalize
            if all(iszero.(X[trial][1:num_neurons[1], t_now+1]))
                X[trial][1:num_neurons[1], t_now+1] = randn(num_neurons[1])
            end
            if all(iszero.(X[trial][num_neurons[1]+1:end, t_now+1]))
                X[trial][num_neurons[1]+1:end, t_now+1] = randn(num_neurons[2])
            end
            X[trial][1:num_neurons[1], t_now+1] ./=
                norm(X[trial][1:num_neurons[1], t_now+1])
            X[trial][num_neurons[1]+1:end, t_now+1] ./
            norm(X[trial][num_neurons[1]+1:end, t_now+1])

            # evolve state according to selected dynamics
            for t in t_now+1:t_now+t_jump
                X[trial][:, t+1] = F * X[trial][:, t]
            end

            # update c matrix
            if selF[1] > 0
                c[trial][selF[1], t_now+1:t_now+t_jump] .= 1
            end
            if selF[2] > 0
                c[trial][size(top_rows, 1)+selF[2], t_now+1:t_now+t_jump] .= 1
            end

            t_now = t_now + t_jump
        end
    end

    return X, c
end

function make_F(
    top_rows::AbstractArray{T,3},
    bot_rows::AbstractArray{T,3},
) where {T<:AbstractFloat}
    num_top = size(top_rows, 1)
    num_bot = size(bot_rows, 1)

    top_opt = rand(1:num_top)
    bot_opt = rand(1:num_bot)

    top_sel = top_rows[top_opt, :, :]
    bot_sel = bot_rows[bot_opt, :, :]

    F = cat(top_sel, bot_sel; dims = 1)

    return F, [top_opt, bot_opt]
end
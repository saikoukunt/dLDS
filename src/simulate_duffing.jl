function simulate_duffing_oscillator(
    num_timepoints::Int,
    num_trials::Int;
    α::T = 1.0,
    β::T = 50.0,
    γ::T = 0.001,
    δ::T = 0.02,
    ω::T = 0.5,
    dt::T = 0.01,
) where {T<:AbstractFloat}
    x = Vector{AbstractMatrix{T}}(undef, num_trials)
    f1 = [1.0 dt; -α*dt 1-δ*dt]
    f2 = [0.0 0.0; -β*dt 0.0]
    for trial in axes(x, 1)
        x[trial] = Matrix{T}(undef, 2, num_timepoints)
        x[trial][:, 1] = 0.5 .* rand(T, 2)

        for t in 2:size(x[trial], 2)
            f_t = f1 + (x[trial][1, t-1]^2) .* f2
            x[trial][:, t] = f_t * x[trial][:, t-1] - [0; γ * cos(ω * t * dt)]
        end
    end

    return x
end
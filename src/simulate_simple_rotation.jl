function simulate_simple_rotation(
	num_timepoints::Int,
	num_trials::Int,
	θ::AbstractFloat = 5;
	min_switch_time::Int = 100,
	max_extra_switch_time::Int = 300,
)
	θ *= 2*π/360

	X = Vector{Matrix{Float64}}(undef, num_trials)
	c = Vector{Matrix{Float64}}(undef, num_trials)
	for trial in 1:num_trials
		X[trial] = zeros(Float64, 2, num_timepoints)
		c[trial] = zeros(Float64, 2, num_timepoints)
	end

	F = Array{Float64, 3}(undef, 2, 2, 2)
	F[1, :, :] = [cos(θ) sin(θ); -sin(θ) cos(θ)]
	F[2, :, :] = [cos(-θ) sin(-θ); -sin(-θ) cos(-θ)]

	for trial in 1:num_trials
		generate_switching_c!(
			c[trial],
			num_timepoints,
			2,
			min_switch_time = min_switch_time,
			max_extra_switch_time = max_extra_switch_time,
		)

		X[:, 1] = randn(T, size(X, 1))
		X[:, 1] /= norm(X[:, 1])

		F_t = Matrix{T}(undef, 2, 2)
		for t in 2:num_timepoints
			fill!(F_t, zero(T))
			for motif in axes(F, 1)
				axpy!(c[motif, t], F[motif, :, :], F_t)
			end

			X[:, t] = F_t * @view(X[:, t-1])
		end
	end

	return X, c, F
end

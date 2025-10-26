using Plots

function plot_Fs(
	F::AbstractArray{T, 3},
	F_hat::AbstractVector{<:AbstractMatrix{T}},
) where {T <: AbstractFloat}
	plots = []
	num_motifs = max(size(F_hat, 1), size(F, 1))
	for i in axes(F, 1)
		push!(
			plots,
			heatmap(F[i, :, :], aspect_ratio = :equal, legend = false, clim = (-1, 1)),
		)
	end
	for i in axes(F_hat, 1)
		push!(
			plots,
			heatmap(
				F_hat[i],
				aspect_ratio = :equal,
				legend = false,
				clim = (-1, 1),
			),
		)
	end
	plot(
		plots...,
		layout = (2, num_motifs),
		showaxis = false,
		size = (200 * num_motifs, 400),
	)
end

function plot_cs(
	c_hat::AbstractMatrix{T},
	c::AbstractArray{T},
) where {T <: AbstractFloat}
	plots = []
	push!(plots, heatmap(c, size = (1000, 250)))
	push!(plots, heatmap(c_hat, size = (1000, 250)))

	plot(plots..., layout = (2, 1))
end

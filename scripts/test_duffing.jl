using Distributed
addprocs(12 - nprocs())

using Plots
@everywhere using DLDS

x = simulate_duffing_oscillator(1000, 50)

plots = []
for trial in 1:4
    push!(
        plots,
        scatter(
            x[trial][1, :],
            x[1][2, :],
            zcolor = 1:1000,
            color = :vanimo,
            markerstrokewidth = 1,
            markersize = 7.5,
        ),
    )
end
plot(plots..., layout = (2, 2), size = (1500, 1000))

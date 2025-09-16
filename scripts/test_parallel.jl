using Distributed
using BenchmarkTools
addprocs(24)

@everywhere using DLDS

# X, c, F = simulate_two_subsystems_no_obs(3000, [4, 4], [3, 3], 50)
X, c, F = matlab_simulation(3000, [4, 4], [3, 3], 50)
num_motifs = 6
num_latents = 8
T = Float64

trial_data = sample_snippets(X, 50, 200)

num_threads = Threads.nthreads()
FX_prod = Array{T,3}(undef, num_latents, num_motifs, num_threads) # for update_c!
FX_prod_gram = Array{T,3}(undef, num_motifs, num_motifs, num_threads)
c_hat = Array{T,3}(undef, size(trial_data, 1), num_motifs, size(trial_data[1], 2) - 1)

function test_c_threaded(
    num_motifs,
    num_latents,
    T,
    trial_data,
    F,
    c,
    c_smooth_coeff,
    c_l1_coeff,
    c_fista_tol,
    c_fista_max_iter,
)
    # solve for c given F

    i = 0
    update_c_threaded!(
        c_hat,
        FX_prod,
        FX_prod_gram,
        trial_data,
        F,
        smooth_coeff = c_smooth_coeff,
        l1_coeff = c_l1_coeff;
        tol = c_fista_tol,
        max_iter = c_fista_max_iter,
        warm_start = false,
    )
end

function test_c_distributed(
    num_motifs,
    num_latents,
    T,
    trial_data,
    F,
    c,
    c_smooth_coeff,
    c_l1_coeff,
    c_fista_tol,
    c_fista_max_iter,
)
    c_hat =
        Array{T,3}(undef, size(trial_data, 1), num_motifs, size(trial_data[1], 2) - 1)
    print(size(c_hat))
    update_c_distributed!(
        c_hat,
        trial_data,
        F,
        smooth_coeff = c_smooth_coeff,
        l1_coeff = c_l1_coeff;
        tol = c_fista_tol,
        max_iter = c_fista_max_iter,
        warm_start = false,
    )
end

@benchmark update_c_threaded!(
    $c_hat,
    $FX_prod,
    $FX_prod_gram,
    $trial_data,
    $F,
    smooth_coeff = 0.4,
    l1_coeff = 0.1;
    tol = 1e-8,
    max_iter = 3000,
    warm_start = false,
)

@benchmark update_c_distributed!(
    $c_hat,
    $trial_data,
    $F,
    smooth_coeff = 0.4,
    l1_coeff = 0.1;
    max_iter = 3000,
    tol = 1e-8,
    warm_start = false,
)

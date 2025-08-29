module DLDS

using LinearAlgebra
using Distributions
using Random
using Statistics
using ProximalOperators
using ProximalAlgorithms
using SparseArrays
using Printf
using GLMNet
using Base.Threads

export fit_full_model, fit_no_obs_model, infer_full_state, infer_no_obs_state
export update_c!, update_D!, update_F!, update_X!
export generate_switching_c!, create_random_dynamics
export simulate_two_subsystems_no_obs

include("./matrix_utils.jl")
include("./simulation_utils.jl")
include("./model.jl")
include("./sample_trials.jl")
include("./fit_infer.jl")
include("./simulate_two_subsystems.jl")

end # module DLDS

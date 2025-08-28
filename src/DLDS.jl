module DLDS

include("./model.jl")
include("./fit_infer.jl")
include("./simulate_multiple_subsystems.jl")

export train_dLDS, update_c!, update_D!, update_F!, update_X!
greet() = print("Hello World!")

end # module DLDS

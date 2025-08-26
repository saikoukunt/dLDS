module DLDS

include("./dlds_model.jl")

export train_dLDS,
    update_c!, update_D!, update_F!, update_X!, init_matrix, normalize_matrix!

greet() = print("Hello World!")

end # module DLDS

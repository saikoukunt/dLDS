using Distributions
using Random

module InitDistribution
abstract type AbstractDistribution end
struct Normal <: AbstractDistribution end
struct Uniform <: AbstractDistribution end
struct DiscreteUniform <: AbstractDistribution end
struct Sparse <: AbstractDistribution end
struct Regional <: AbstractDistribution end
end

"""
    init_matrix(distribution, matrix_size, random_seed; kwargs...)

Initializes a matrix of a given size with a specified distribution.

# Arguments
- `distribution::InitDistribution.AbstractDistribution`: A distribution type tag (e.g., `InitDistribution.Normal()`).
- `matrix_size::Tuple{Vararg{Int}}`: The dimensions of the matrix.
- `random_seed::Int`: The seed for the random number generator.

# Keyword Arguments
- `normalize::Bool=false`: If true, the matrix is normalized after creation.
- Other `kwargs...`: Distribution-specific parameters (e.g., `loc`, `scale`, `k`).
"""
function init_matrix(
    distribution::InitDistribution.AbstractDistribution,
    matrix_size::Tuple{Vararg{Int}},
    random_seed::Int;
    normalize::Bool = false,
    kwargs...,
)
    Random.seed!(random_seed)
    random_matrix = _init_matrix(distribution, matrix_size; kwargs...)

    if normalize
        return normalize_matrix!(random_matrix)
    end
    return random_matrix
end

_init_matrix(
    distribution::InitDistribution.Normal,
    matrix_size::Tuple{Vararg{Int}};
    loc::Float64 = 0.0,
    scale::Float64 = 1.0,
) = rand(Normal(loc, scale), matrix_size)

_init_matrix(
    distribution::InitDistribution.Uniform,
    matrix_size::Tuple{Vararg{Int}};
    low::Float64 = 0.0,
    high::Float64 = 1.0,
) = rand(Uniform(low, high), matrix_size)

_init_matrix(
    distribution::InitDistribution.DiscreteUniform,
    matrix_size::Tuple{Vararg{Int}};
    low::Int = 0,
    high::Int = 10,
) = Float64.(rand(low:high, matrix_size))

function _init_matrix(
    distribution::InitDistribution.Sparse,
    matrix_size::Tuple{Vararg{Int}};
    k::Int,
)
    rows, cols = matrix_size
    random_mat = zeros(rows, cols)
    for col in 1:cols
        num_nonzero = rand(1:min(k, rows))
        rows_indices = rand(1:rows, num_nonzero)
        random_mat[rows_indices, col] = 1
    end

    return random_mat
end

# REVIEW: This might need to handle 3D array
# NOTE: removed exp normalization for now since it can't be done in place
"""
    normalize_matrix!(A, norm_type=:eigen)

Applies the specified normalization to the matrix `A` in-place.

# Arguments
- `A::AbstractMatrix`: The matrix to normalize.
- `norm_type::Symbol`: The type of normalization to apply. Options are:
    - `:eigen`: Normalize by the largest eigenvalue.
    - `:max`: Normalize by the maximum absolute value.
    - `:exp`: Normalize using the matrix exponential.
"""
function normalize_matrix!(
    A::AbstractMatrix{<:AbstractFloat},
    norm_type::Symbol = :eigen,
)
    if norm_type == :eigen
        norm_factor = maximum(abs.(eigvals(A)))
    elseif norm_type == :max
        norm_factor = maximum(abs.(A))
    else
        error("Unknown normalization type: $norm_type. Use :eigen, :max, or :exp.")
    end

    A ./= norm_factor
    return A
end
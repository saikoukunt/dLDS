function sample_snippets(
    data::AbstractArray{T,3},
    num_snippets::Int,
    samples_per_snippet::Int,
) where {T<:AbstractFloat}
    snippets = Vector{Matrix{Float64}}(undef, num_snippets)

    trial_inds =
        num_snippets == size(data, 1) ? (1:num_snippets) :
        rand(1:size(data, 1), num_snippets)

    for (i, trial_id) in enumerate(trial_inds)
        trial_length = size(data[trial_id, :, :], 2)
        if trial_length <= samples_per_snippet
            snippets[i] = @view(data[trial_id, :, :])
        else
            t_start = rand(1:(trial_length-samples_per_snippet))
            t_end = t_start + samples_per_snippet - 1
            snippets[i] = @view(data[trial_id, :, t_start:t_end])
        end
    end

    return snippets
end

function sample_snippets_weighted() end
"""
Slice sampler based on
[Neal, 2003](https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full).
"""
@kwdef @concrete struct SliceSampler
    w = 1.0 # initial slice size
    p = 10 # slices are no larger than 2^p * w
    dim_fraction = 1.0 # proportion of variables to update
end


"""
$SIGNATURES 
"""
@provides explorer create_explorer(target, inputs) = SliceSampler() # TODO
create_state_initializer(target) = Ref(zeros(target)) # TODO
adapt_explorer(explorer::SliceSampler, _, _) = explorer 
explorer_recorder_builders(::SliceSampler) = [] 
regenerate!(explorer::SliceSampler, replica, shared) = @abstract # TODO or remove

function step!(explorer::SliceSampler, replica, shared)
    log_potential = find_log_potential(replica, shared)
    slice_sample!(explorer, replica.state, log_potential)
end


"""
$SIGNATURES
Slice sample one point.
"""
function slice_sample!(h::SliceSampler, state::AbstractVector, log_potential)
    g_x0 = -log_potential(state) # TODO: is it correct to keep the vertical draw out of the loop?
    for c in 1:length(state) # update *every* coordinate (TODO: change this later!)
        pointer = Ref(state, c)
        slice_sample_coord!(h, state, pointer, log_potential, g_x0)
    end
end

function slice_sample!(h::SliceSampler, state::DynamicPPL.TypedVarInfo, log_potential)
    transform_back = false
    if !DynamicPPL.istrans(state, DynamicPPL._getvns(state, DynamicPPL.SampleFromPrior())[1]) # check if in constrained space
        DynamicPPL.link!(state, DynamicPPL.SampleFromPrior()) # transform to unconstrained space
        transform_back = true # transform it back after log_potential evaluation
    end
    g_x0 = -log_potential(state)
    for i in 1:length(keys(state.metadata))
        for c in 1:length(state.metadata[i].vals)
            pointer = Ref(state.metadata[i].vals, c)
            slice_sample_coord!(h, state, pointer, log_potential, g_x0)
        end
    end
    if transform_back
        DynamicPPL.invlink!!(state, log_potential.model) # transform back to constrained space
    end
end

function slice_sample_coord!(h, state, pointer, log_potential, g_x0)
    z = g_x0 - rand(Exponential(1.0)) # log(vertical draw)
    L, R = slice_double(h, state, z, pointer, log_potential)
    pointer[] = slice_shrink(h, state, z, L, R, pointer, log_potential)
end


"""
$SIGNATURES
Double the current slice.
"""
function slice_double(h::SliceSampler, state, z, pointer, log_potential)
    old_position = pointer[] # store old position (trick to avoid memory allocation)
    U = rand()
    L = old_position - h.w*U # new left endpoint
    R = L + h.w
    K = h.p
    
    pointer[] = L
    neg_potent_L = -log_potential(state) # store the negative log potential
    pointer[] = R
    neg_potent_R = -log_potential(state)

    while (K > 0) && ((z < neg_potent_L) || (z < neg_potent_R))
        V = rand()        
        if V <= 0.5
            L = L - (R - L)
            pointer[] = L
            neg_potent_L = -log_potential(state) # store the new neg log potential
        else
            R = R + (R - L)
            pointer[] = R
            neg_potent_R = -log_potential(state)
        end
        K = K - 1
    end
    pointer[] = old_position # return the state back to where it was before
    return(; L, R)
end


"""
$SIGNATURES
Shrink the current slice.
"""
function slice_shrink(h::SliceSampler, state, z, L, R, pointer, log_potential)
    old_position = pointer[]
    Lbar = L
    Rbar = R

    while true
        U = rand()
        new_position = Lbar + U * (Rbar - Lbar)
        pointer[] = new_position 
        consider = (z < -log_potential(state))
        pointer[] = old_position
        if (consider) && (slice_accept(h, state, new_position, z, L, R, pointer, log_potential))
            return new_position
        end
        if new_position < pointer[]
            Lbar = new_position
        else
            Rbar = new_position
        end
    end
    return new_position
end


"""
$SIGNATURES
Test whether to accept the current slice.
"""
function slice_accept(h::SliceSampler, state, new_position, z, L, R, pointer, log_potential)
    old_position = pointer[]
    Lhat = L
    Rhat = R

    pointer[] = L # trick to avoid memory allocation
    neg_potent_L = -log_potential(state)
    pointer[] = R 
    neg_potent_R = -log_potential(state)
    
    D = false
    acceptable = true
    
    while Rhat - Lhat > 1.1 * h.w
        M = (Lhat + Rhat)/2.0
        if ((old_position < M) && (new_position >= M)) || ((old_position >= M) && (new_position < M))
            D = true
        end
        
        if new_position < M
            Rhat = M
            pointer[] = Rhat
            neg_potent_R = -log_potential(state)
        else
            Lhat = M
            pointer[] = Lhat
            neg_potent_L = -log_potential(state)
        end
        
        if (D && (z >= neg_potent_L) && (z >= neg_potent_R))
            pointer[] = old_position 
            return false
        end
    end
    pointer[] = old_position
    return acceptable
end
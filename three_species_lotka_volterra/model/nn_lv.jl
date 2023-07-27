function create_model(act_fct_name, hidden_layers, hidden_neurons, p_ph, n_out)
    if act_fct_name == "tanh"
        act_fct = Lux.tanh
    elseif act_fct_name == "gelu"
        act_fct = Lux.gelu
    elseif act_fct_name == "rbf"
        act_fct = x -> Lux.exp.(-(x.^2))
    elseif act_fct_name == "identity"
        act_fct = x -> x
    end

    # Construct the Neural Network Component of the UDE
    first_layer = Dense(3, hidden_neurons, act_fct)
    intermediate_layers = [Dense(hidden_neurons, hidden_neurons, act_fct) for _ in 1:hidden_layers-1]
    last_layer = Dense(hidden_neurons, n_out, init_weight = Lux.zeros32
    )
    augmented_dynamics = Lux.Chain(first_layer, intermediate_layers..., last_layer)

    # Setup parameters of NN
    rng = Random.default_rng()
    Random.seed!(rng, 1)
    ps, st = Lux.setup(rng, augmented_dynamics)
        
    # Combine all learnable parameters (i.e. NN parameters and physical parameters)
    ps = merge(p_ph, (ude = ps,))
    ps = ComponentArray(ps) 
    st = st

    return augmented_dynamics, ps, st
end
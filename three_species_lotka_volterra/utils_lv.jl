##################### Model utils ###############################

"""
Given parameters in the untransformed space, transform them to the transformed space 

#### Arguments:
    par:        parameter value
    lb:         lower bound on parameter
    ub:         upper bound on parameter
    transform:  parameter transform used, must be one of
                - "tanh_bounds": tanh, shifted and scaled to fit lb and ub
                - "log": log transform without bounds
                - "identity": no transform
"""
function transform_par(par; lb::Float64 = param_range[1], ub::Float64 = param_range[2], transform=transform::String)
    if transform == "tanh_bounds"
        return atanh(2*(par-lb)/(ub-lb)-1) + 5.74
    elseif transform == "log"
        return log(par)
    elseif transform == "identity"
        return par
    end
end

"""
Given parameters in the transformed space, transform them back to the parameter's actual values 

#### Arguments:
    par:        parameter value
    lb:         lower bound on parameter
    ub:         upper bound on parameter
    transform:  parameter transform used, must be one of
                - "tanh_bounds": tanh, shifted and scaled to fit lb and ub
                - "log": log transform without bounds
                - "identity": no transform
"""
function inverse_transform(par, lb::Float64 = param_range[1], ub::Float64 = param_range[2], transform = transform::String)
    if transform == "tanh_bounds"
        return lb + (Lux.tanh(par-5.74) + 1)/2*(ub-lb)
    elseif transform == "log"
        return Lux.exp(par)
    elseif transform == "identity"
        return par
    end
end;

######################## Training Helper Functions ######################


"""
    nll(θ, IC::Vector{Float64}=IC, solver=solver, t::Vector{Float64}=t, y_obs::Matrix{Float64}=y_obs; st_nn::NamedTuple = st, nn_model::Chain = nn_model)

Calculates the loss based on the model's predictions of observed states only (i.e. no regularization).

#### Arguments
    - θ:            parameters of the model to be optimized
    - model_noise:  bool, defines the loss function
                    if true: loss is equivalent to negative log likelihood
                    if false: loss is equivalent to the squared error
"""
function nll(θ, IC::Vector{Float64}=IC, solver=solver, t::Vector{Float64}=t, y_obs::Matrix{Float64}=y_obs; st_nn::NamedTuple = st, nn_model::Chain = nn_model) #y_hidden::Matrix{Float64}=0
    l = convert(eltype(θ), 0)
    # solve ODE
    _prob = remake(prob_nn, u0 = IC, tspan = (t[1], t[end]), p = θ)  # update ODE Problem with nn parameters
    tmp_sol = solve(_prob, solver, saveat = t,
                abstol=tolerance, reltol=tolerance,
                sensealg = sense
                )
    if size(tmp_sol) == size(y_obs) # see https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/training_tips/divergence/
        @inbounds X̂ = Array(tmp_sol)
        #loss = n_t * log(sigma) + 1/sigma² * sum_t (pred-true)² = lc1 + lc2
        log_sigma_sq = [θ.n_u1, θ.n_u2, θ.n_u3]
        lc_1 = size(y_obs)[2]/2 * log_sigma_sq
        lc_2 = sum(abs2, X̂ .- y_obs, dims=2) ./ exp.(log_sigma_sq)*0.5
        return sum(lc_1+lc_2)  
    else
        return Inf
    end
end;


function train_lv(p::ComponentVector, st::NamedTuple, lr_adam::Float64, λ_reg::Float64, stepnorm_bfgs::Float64, epochs::Tuple{Int, Int}, l_mech::Int)
    # Container to track the training
    losses = Float64[];
    losses_regularization = Float64[];
    r1 = Float64[];
    a1_1 = Float64[];
    a1_2 = Float64[];
    a1_3 = Float64[];
    r2 = Float64[];
    a2_1 = Float64[];
    a2_2 = Float64[];
    a2_3 = Float64[];
    r3 = Float64[];
    a3_1 = Float64[];
    a3_2 = Float64[];
    a3_3 = Float64[];


    l_nn = length(p) - l_mech # parameter total - interaction parameters - noise parameters

    callback = function (p, l)
        @ignore_derivatives push!(losses, l)
        @ignore_derivatives push!(r1, @inbounds p[1])
        @ignore_derivatives push!(a1_1, @inbounds p[2])
        @ignore_derivatives push!(a1_2, @inbounds p[3])
        @ignore_derivatives push!(a1_3, @inbounds p[4])
        @ignore_derivatives push!(r2, @inbounds p[5])
        @ignore_derivatives push!(a2_1, @inbounds p[6])
        @ignore_derivatives push!(a2_2, @inbounds p[7])
        @ignore_derivatives push!(a2_3, @inbounds p[8])
        @ignore_derivatives push!(r3, @inbounds p[9])
        @ignore_derivatives push!(a3_1, @inbounds p[10])
        @ignore_derivatives push!(a3_2, @inbounds p[11])
        @ignore_derivatives push!(a3_3, @inbounds p[12])
        
        @ignore_derivatives push!(losses_regularization, sum(abs2, @inbounds p[l_mech+1:end])/l_nn)
        
        if length(losses)%5==0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end;

    loss(θ) = nll(θ) + convert(eltype(θ), λ_reg)*sum(abs2, @inbounds θ[l_mech+1:end])./l_nn

    # First train with ADAM for better convergence -> move the parameters into a
    # favourable starting positing for BFGS
    adtype = Optimization.AutoForwardDiff();
    optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype); # x: state variables, p: hyperparameters
    optprob = Optimization.OptimizationProblem(optf, p);
    res1 = Optimization.solve(optprob, ADAM(lr_adam), callback=callback, maxiters = epochs[1])
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer);
    res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=stepnorm_bfgs), callback=callback, maxiters = epochs[2])
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    return res2.u, st, losses, losses_regularization, r1, a1_1, a1_2, a1_3, r2, a2_1, a2_2, a2_3, r3, a3_1, a3_2, a3_3
end

########## Plotting utils ###############
function plot_loss_trajectory(losses::Vector{Float64}; path_to_store::String="", return_plot::Bool=false)
    pl_losses = plot(1:length(losses), losses, xlabel = "Epoch", ylabel = "Loss", title="Loss Curve", legend=false)
    if return_plot
        return pl_losses
    else
        savefig(pl_losses, joinpath(path_to_store, "loss_curve.png")) 
    end
end

function plot_regularization_loss_trajectory(losses::Vector{Float64}; path_to_store::String="", return_plot::Bool=false)
    pl_losses = plot(1:length(losses), losses, xlabel = "Epoch", ylabel = "Regularization Strength", title="Loss Curve", legend=false)
    if return_plot
        return pl_losses
    else
        savefig(pl_losses, joinpath(path_to_store, "regularization_loss_curve.png")) 
    end
end

function plot_parameter_trajectory(parameter_estimates::Vector{Float64}, true_value::Float64, parameter_name::String; path_to_store::String="", return_plot::Bool=false)
    par_plot = plot(1:length(parameter_estimates), inverse_transform.(parameter_estimates),  yaxis = :log10, xlabel="Epoch", ylabel="Parameter Value", label="Estimation", title=parameter_name)
    plot!(1:length(parameter_estimates), Base.repeat([true_value], length(parameter_estimates)), yaxis = :log10, label="True Value",  xlabel="Epoch", ylabel="Parameter Value", title=parameter_name, line=:dash)
    if return_plot
        return par_plot
    else
        savefig(par_plot, joinpath(path_to_store, "$(parameter_name).png"))
    end
end

function plot_observed_trajectory(observable_name::String, t_pred::Vector{Float64}, y_pred::Vector{Float64}, t_full::Vector{Float64}, y_full::Vector{Float64}, t_obs::Vector{Float64}, y_obs::Vector{Float64}; std::Float64 = 0.0, path_to_store::String="", return_plot::Bool=false, plot_observed::Bool=true, plot_noise::Bool=true)
    plot_pSTAT5A_rel = plot(t_full, y_full, 
        label="simulated", xlabel="time", ylabel=observable_name, line=:dash, color=2, margin=6mm)
    if plot_observed
        scatter!(t_obs, y_obs, 
            label="observed data", xlabel="time", ylabel=observable_name, markershape=:x, color=2, margin = 6mm)
    end
    if plot_noise
        plot!(t_pred, y_pred, 
            xlabel="time", ylabel=observable_name, label="prediction", title=observable_name, color=1, margin = 6mm, ribbon = std)
    else
        plot!(t_pred, y_pred, 
            xlabel="time", ylabel=observable_name, label="prediction", title=observable_name, color=1, margin = 6mm, ribbon = std)
    end
    if return_plot
        return plot_pSTAT5A_rel
    else
        savefig(par_plot, joinpath(path_to_store, "$(observable_name).png"))
    end
end

function plot_hidden_trajectory(state_name::String, t_pred::Vector{Float64}, y_pred::Vector{Float64}, t_full::Vector{Float64}, y_full::Vector{Float64}; std::Float64 = 0.0, path_to_store::String="", return_plot::Bool=false)
    plot_pSTAT5A_rel = plot(t_full, y_full, 
        label="simulated", xlabel="time", ylabel=state_name, line=:dash, color=2, margin=6mm)
    plot!(t_pred, y_pred, 
            xlabel="time", ylabel=state_name, label="prediction", title=state_name, color=1, margin = 8mm, ribbon = std)
    if return_plot
        return plot_pSTAT5A_rel
    else
        savefig(par_plot, joinpath(path_to_store, "$(state_name).png"))
    end
end

### plotting utils specific for the Boehm problem

function plot_hidden_boehm(t_plot::Vector{Float64}, pred::Matrix{Float64}, t_full::Vector{Float64}, y_hidden_full::Matrix{Float64}, path_to_store::String)
    p1 = plot_hidden_trajectory("STAT5A", t_plot, pred[1,:], t_full, y_hidden_full[1,:]; return_plot=true)
    p2= plot_hidden_trajectory("STAT5B", t_plot, pred[2,:], t_full, y_hidden_full[2,:]; return_plot=true)
    p3 = plot_hidden_trajectory("pApB", t_plot, pred[3,:], t_full, y_hidden_full[3,:]; return_plot=true)
    p4 = plot_hidden_trajectory("pApA", t_plot, pred[4,:], t_full, y_hidden_full[4,:]; return_plot=true)
    p5 = plot_hidden_trajectory("pBpB", t_plot, pred[5,:], t_full, y_hidden_full[5,:]; return_plot=true)
    p6 = plot_hidden_trajectory("nucpApA", t_plot, pred[6,:], t_full, y_hidden_full[6,:]; return_plot=true)
    p7 = plot_hidden_trajectory("nucpApB", t_plot, pred[7,:], t_full, y_hidden_full[7,:]; return_plot=true)
    p8 = plot_hidden_trajectory("nucpBpB", t_plot, pred[8,:], t_full, y_hidden_full[8,:]; return_plot=true)
    prediction_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(2, 4 ), size=(1700, 800))
    savefig(prediction_plot, joinpath(path_to_store, "hidden_states.png"))
end

function plot_observed_lv(t_plot::Vector{Float64}, pred_obs::Matrix{Float64}, t_full::Vector{Float64}, y_obs_full::Matrix{Float64}, t::Vector{Float64}, y_obs::Matrix{Float64}, path_to_store::String, ps::ComponentVector)
    p1 = plot_observed_trajectory("u1", t_plot, pred_obs[1,:], t_full, y_obs_full[1,:], t, y_obs[1,:]; std=sqrt.(exp.(ps.n_u1)), return_plot=true, plot_observed=true, plot_noise=true)
    p2 = plot_observed_trajectory("u2", t_plot, pred_obs[2,:], t_full, y_obs_full[2,:], t, y_obs[2,:]; std=sqrt.(exp.(ps.n_u2)), return_plot=true, plot_observed=true, plot_noise=true)
    p3 = plot_observed_trajectory("u3", t_plot, pred_obs[3,:], t_full, y_obs_full[3,:], t, y_obs[3,:]; std=sqrt.(exp.(ps.n_u3)), return_plot=true, plot_observed=true, plot_noise=true)
    prediction_plot = plot(p1, p2, p3, layout=(1, 3), size=(1300, 400))
    savefig(prediction_plot, joinpath(path_to_store, "observables_with_noise.png"))
    
    p1 = plot_observed_trajectory("u1", t_plot, pred_obs[1,:], t_full, y_obs_full[1,:], t, y_obs[1,:]; return_plot=true, plot_observed=true, plot_noise=false)
    p2 = plot_observed_trajectory("u2", t_plot, pred_obs[2,:], t_full, y_obs_full[2,:], t, y_obs[2,:]; return_plot=true, plot_observed=true, plot_noise=false)
    p3 = plot_observed_trajectory("u3", t_plot, pred_obs[3,:], t_full, y_obs_full[3,:], t, y_obs[3,:]; return_plot=true, plot_observed=true, plot_noise=false)
    prediction_plot = plot(p1, p2, p3, layout=(1, 3), size=(1300, 400))
    savefig(prediction_plot, joinpath(path_to_store, "observables_without_noise.png"))
end

########### Data storage utils ###########
function create_summary_metrics_file(path::String)
    open(joinpath(path, "summary.csv"), "w") do io
        writedlm(io, ["problem" "mechanistic_setting" "dataset" "sampling_strategy" "parameter_row" "array_nr" "epochs_adam" "epochs_bfgs" "lr_adam" "stepnorm_bfgs" "λ_reg" "act_fct_name" "hidden_layers" "hidden_neurons" "tolerance" "hidden_MSE" "hidden_nMSE" "obs_MSE" "obs_nMSE" "time" "loss" "NegLL"], ",")    
    end
end

function store_parameters(path_to_store::String, parameter_list::Vector{String}, estimate::Vector)
    open(joinpath(path_to_store, "mechanistic_parameters.csv"), "w") do io
        i = 1
        writedlm(io, ["name" "estimate"])
        for par in parameter_list
            writedlm(io, [par estimate[i]])
            i+=1
        end
    end
end

########### Evaluation utils #############
function MSE(y::Matrix{Float64}, y_pred::Matrix{Float64})
    mse = mean((y-y_pred)[:,2:end].^2, dims=2)
    print("n states: $(length(mse))")
    return mse
end

function nMSE(y::Matrix{Float64}, y_pred::Matrix{Float64})
    return mean(((y-y_pred)[:,2:end].^2)./y[:,2:end], dims=2)
end


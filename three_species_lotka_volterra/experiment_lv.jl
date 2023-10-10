############# Import Packages ###################
using CSV, DataFrames
using Lux
using ComponentArrays, SciMLSensitivity, NNlib, Optimisers, OrdinaryDiffEq, Random,
      Statistics, OneHotArrays, StatsBase
using Plots, Measures
using Random
using Dates
using DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Serialization
using ChainRulesCore
using Zygote
using JLD, JLD2
rng = Random.default_rng()
Random.seed!(rng, 1)


############# Experimental Settings ###################
const experiment_name = "05_10_23"

const test_setup = true  # if used on the cluster this has to be set to false
const create_plots = true
one_observable = true


const problem_name = "three_species_lotka_volterra"
const transform = "log";
exp_sampling_strategy = ("no_sampling", )
exp_mechanistic_setting = ("lv_missing_dynamics", )

exp_early_stopping = (5, 10, 15, )
exp_reg = ("l2", "l1", )
exp_λ_reg = (1e-1, 1e-2, 1e-3, )
epochs = (500, 1000) # (epochs_adam, epochs_bfgs)
exp_lr_adam = (1e-3, 1e-2) # lr_bfgs = 0.1*lr_adam
exp_hidden_layers = (1, 2, 3 )
exp_hidden_neurons = (4, 8, )
exp_act_fct = ("tanh", ) 
exp_tolerance = (1e-8, )
exp_par_setting = (1, ) # define what rows of the startpoints.csv file to try out
exp_dataset = ("lotka_volterra_datapoints_40_noise_5", "lotka_volterra_datapoints_80_noise_15", "lotka_volterra_datapoints_80_noise_5", "lotka_volterra_datapoints_40_noise_15")

experiments = collect(Iterators.product(exp_mechanistic_setting, exp_sampling_strategy, exp_dataset, exp_λ_reg, exp_lr_adam, 
    exp_hidden_layers, exp_hidden_neurons, exp_act_fct, exp_tolerance, exp_par_setting, exp_early_stopping, exp_reg));

if test_setup
    array_nr = 1
else 
    array_nr = parse(Int, ARGS[1])
end
skip_step = 5       #until time_break every 5th value is chosen for the validation set
time_break_in_sec = 7.0

mechanistic_setting, sampling_strategy, dataset, λ_reg, lr_adam, hidden_layers, hidden_neurons, act_fct_name, tolerance, par_row, early_stopping, reg = experiments[array_nr]
exp_specifics = array_nr

############# Prepare Experiment #######################
# Load functinalities
if test_setup
    epochs = (500, 500)
    include("$(problem_name)/create_directories_lv.jl")
    include("$(problem_name)/utils_lv.jl")
    include("$(problem_name)/reference.jl")
    include("$(problem_name)/model/$(mechanistic_setting).jl")
    include("$(problem_name)/model/nn_lv.jl")
else
    include("create_directories_lv.jl")
    include("utils_lv.jl")
    include("model/$(mechanistic_setting).jl")
    include("model/nn_lv.jl")
    include("reference.jl")
end

# Define paths
experiment_series_path, experiment_run_path, data_path, parameter_path = create_paths(problem_name, experiment_name, sampling_strategy, "$(par_row)", mechanistic_setting, dataset, "$(array_nr)")

if array_nr == 1
    open(joinpath(experiment_series_path, "summary.csv"), "a") do io
        header = ["problem_name" "mechanistic_setting" "dataset" "sampling_strategy" "par_row" "array_nr" "early stopping" "reg" "epochs_adam" "epochs_bfgs" "lr_adam" "stepnorm_bfgs" "λ_reg" "act_fct" "hidden_layers" "hidden_neurons" "tolerance" "MSE" "nMSE" "runtime" "loss" "negLL" "validation_loss"]
        writedlm(io, header, ",")
    end
    save(joinpath(experiment_series_path, "hp_settings.jld"), 
        "sampling_strategy", exp_sampling_strategy,
        "mechanistic_setting", exp_mechanistic_setting,
        "λ_reg", exp_λ_reg,
        "lr_adam", exp_lr_adam,
        "hidden_layer", exp_hidden_layers,
        "hidden_neurons", exp_hidden_neurons,
        "act_fct", exp_act_fct,
        "tolerance", exp_tolerance,
        "par_setting", exp_par_setting,
        "dataset", exp_dataset,
        "one_observable", one_observable,
        "early_stopping", exp_early_stopping)
    # hp_settings = load(joinpath(experiment_series_path, "hp_settings.jld"))
end

IC, tspan, t, y_obs, t_full, y_obs_full, p_true, p_ph = load_data(data_path, problem_name)

#########training and validation data###############
#find index belonging to the chosen time
time_break = findall(x->x==time_break_in_sec,t)

t_half_1 = t[1:time_break]
t_half_2 = t[time_break:end]

#time points for training
t_train = t_half_1[Not(1:skip_step:end)]
pushfirst!(t_train,0.0)

#time points for validation
t_val = vcat(t_half_1[1:skip_step:end], t_half_2)

#training data values
y_half_1 = y_obs[:,1:time_break]
y_half_2 = y_obs[:,time_break:end]
y_train = [y_half_1[:,1] y_half_1[:,Not(1:skip_step:end)]]

#validation values
y_validation = [y_half_1[:,1:skip_step:end] y_half_2]

# create model 
nn_model, ps, st = create_model(act_fct_name, hidden_layers, hidden_neurons, p_ph, n_out)


dynamics!(du, u, ps, t) = ude_dynamics!(du, u, ps, st, t; nn_model)

# Define the problem
prob_nn = ODEProblem(dynamics!, IC, tspan, ps);

#Training
stepnorm_bfgs = 0.1*lr_adam

# Define a predictor
function predict(θ, X = IC, T = t; tolerance=tolerance, prob_nn=prob_nn)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)  # update ODE Problem with nn parameters
    Array(solve(_prob, saveat = T,
                abstol=tolerance, reltol=tolerance
                ))
end;

l_mech = length(parameter_names)

t1 = now()
p_opt, st, losses, losses_regularization,validation_loss, r1, a1_1, a1_2, a1_3, r2, a2_1, a2_2, a2_3, r3, a3_1, a3_2, a3_3 = train_lv(ps, st, lr_adam, λ_reg, stepnorm_bfgs, epochs, l_mech)
runtime = (now()-t1).value/1000/60
t_plot = t_full
pred = predict(p_opt, IC, t_full)


############### Evaluate Experiment ###################
# log results
# loss curve
if create_plots
    plot_loss_trajectory(losses; path_to_store=experiment_run_path, return_plot=false)
    plot_regularization_loss_trajectory(losses_regularization; path_to_store=experiment_run_path, return_plot=false)
    plot_observed_lv(t_full, pred, t_full, y_obs_full, t, y_obs, experiment_run_path, p_opt )
    plot_val_loss_trajectory(validation_loss; path_to_store=experiment_run_path, return_plot=false)
    plot_val_loss_a_loss(losses, validation_loss, path_to_store=experiment_run_path, return_plot=false)
end

#store training results over epochs
open(joinpath(experiment_run_path, "training_curves.csv"), "w") do io
    header = ["epoch" "loss" "regularization_loss" "validation_loss" "r1" "a1_1" "a1_2" "a1_3" "r2" "a2_1" "a2_2" "a2_3" "r3" "a3_1" "a3_2" "a3_3"]
    writedlm(io, header, ",")
    if mechanistic_setting == "lv_missing_dynamics"
        writedlm(io, [1:length(losses) losses losses_regularization validation_loss repeat([missing], length(losses)) a1_1 a1_2 a1_3 repeat([missing], length(losses)) a2_1 a2_2 a2_3 repeat([missing], length(losses)) a3_1 a3_2 a3_3], ",")
    else
        writedlm(io, [1:length(losses) losses losses_regularization validation_loss r1 a1_1 a1_2 a1_3 r2 a2_1 a2_2 a2_3 r3 a3_1 a3_2 a3_3], ",")
    end
end
# store final values of all mechanistic parameters to csv
pars = vcat(parameter_names)
p_mech = convert(Vector{Union{Missing,Float64}}, p_opt[1:length(pars)])
save(joinpath(experiment_run_path, "nn_settings.jld"), 
    "r1", p_opt.r1,
    "a1_1", p_opt.a1_1,
    "a1_2",p_opt.a1_2,
    "a1_3", p_opt.a1_3,
    "r2", p_opt.r2,
    "a2_1", p_opt.a2_1,
    "a2_2",p_opt.a2_2,
    "a2_3", p_opt.a2_3,
    "r3", p_opt.r3,
    "a3_1", p_opt.a3_1,
    "a3_2",p_opt.a3_2,
    "a3_3", p_opt.a3_3,
    "n_u1", p_opt.n_u1,
    "n_u2", p_opt.n_u2,
    "n_u3", p_opt.n_u3,
    "ude", convert(Vector{Union{Missing,Float64}},p_opt.ude))
if mechanistic_setting == "lv_missing_dynamics"
    p_mech[1] = missing
    p_mech[5] = missing
    p_mech[9] = missing
end
#store parameters
store_parameters(experiment_run_path, pars, p_mech)


# store model predictions
open(joinpath(experiment_run_path, "predictions.csv"), "w") do io
    header = ["t" "u1" "u2" "u3"]
    writedlm(io, header, ",")
    writedlm(io, [t_full pred[1,:] pred[2,:] pred[3,:]], ",")
end

#hidden_MSE = MSE(y_hidden, pred)
#hidden_nMSE = nMSE(y_hidden, pred)
obs_MSE = MSE(y_obs_full, pred)
obs_nMSE = nMSE(y_obs_full, pred)

NegLL = nll(p_opt)

open(joinpath(experiment_series_path, "summary.csv"), "a") do io
    loss = losses[end]
    val_loss = validation_loss[end]
    writedlm(io, [problem_name mechanistic_setting dataset sampling_strategy par_row array_nr early_stopping reg epochs[1] epochs[2] lr_adam stepnorm_bfgs λ_reg act_fct_name hidden_layers hidden_neurons tolerance mean(obs_MSE) mean(obs_nMSE) runtime loss NegLL val_loss], ",")    
end
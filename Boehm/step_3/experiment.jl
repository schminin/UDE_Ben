############# Import Packages ###################
using Lux
using ComponentArrays, SciMLSensitivity, NNlib, Optimisers, OrdinaryDiffEq, Random,
      Statistics, OneHotArrays
using Plots, Measures
using Random
using Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Serialization
using ChainRulesCore
using ForwardDiff
rng = Random.default_rng()
Random.seed!(rng, 1)

############ Define Experiment Series #################
const test_setup = false  # if used on the cluster this has to be set to false
const create_plots = false

const experiment_name = "05_05_2023"

const transform = "tanh_bounds";
const param_range = (1e-5* (1-1e-6), 100000.0 * (1+1e-6));

const problem_name = "Boehm"
exp_mechanistic_setting = ("boehm_fully_known", "boehm_missing_interaction", "boehm_missing_state", "boehm_missing_observable_pSTAT5A_rel")
exp_sampling_strategy = ("optimization_endpoints", ) # ("latin_hypercube_sampling", "optimization_endpoints")
exp_dataset = ("original", ) #("original", "synthetic_datapoints_32_noise_0", "synthetic_datapoints_32_noise_5", "synthetic_datapoints_32_noise_20") # ("original",) # 

exp_λ_reg = (1e4, 1e3, 1e2, 1.0, 1e-2, 1e-3, )
epochs = (500, 3000) # (epochs_adam, epochs_bfgs)
exp_lr_adam = (1e-4, 1e-3, 1e-2, 1e-1) # lr_bfgs = 0.1*lr_adam
exp_hidden_layers = (1, 2, 3, )
exp_hidden_neurons = (4, 8, 16)
exp_act_fct = ("tanh", ) # identity
exp_tolerance = (1e-12, )
exp_par_setting = (1, ) # define what rows of the startpoints.csv file to try out
const experiments = collect(Iterators.product(exp_mechanistic_setting, exp_sampling_strategy, exp_dataset, exp_λ_reg, exp_lr_adam, 
    exp_hidden_layers, exp_hidden_neurons, exp_act_fct, exp_tolerance, exp_par_setting));

const solver = KenCarp4()
const sense = ForwardDiffSensitivity()

if test_setup
    array_nr = 1
else 
    array_nr = parse(Int, ARGS[1])
end

mechanistic_setting, sampling_strategy, dataset, λ_reg, lr_adam, hidden_layers, hidden_neurons, act_fct_name, tolerance, par_row = experiments[array_nr]
exp_specifics = array_nr

############# Prepare Experiment #######################
# Load functinalities
if test_setup
    epochs = (10, 10)
    include("$(problem_name)/step_3/create_directories.jl")
    include("$(problem_name)/step_3/utils.jl")
    include("$(problem_name)/step_3/reference.jl")
    include("$(problem_name)/step_3/model/$(mechanistic_setting).jl")
    include("$(problem_name)/step_3/model/nn.jl")
else
    include("create_directories.jl")
    include("utils.jl")
    include("model/$(mechanistic_setting).jl")
    include("model/nn.jl")
    include("reference.jl")
end

# Define paths
experiment_series_path, experiment_run_path, data_path, parameter_path = create_base_directories(problem_name, experiment_name, sampling_strategy, "$(par_row)", mechanistic_setting, dataset, "$(array_nr)")

# prepare storage of results
if array_nr == 1
    create_summary_metrics_file(experiment_series_path)
end

# Load data
IC, tspan, t, y_obs, y_hidden, t_full, y_obs_full, y_hidden_full, p_true, p_ph = load_data(data_path, parameter_path, problem_name, par_row)

# create model 
nn_model, ps, st = create_model(act_fct_name, hidden_layers, hidden_neurons, p_ph, n_out)

############### Run Experiment ########################
stepnorm_bfgs = 0.1*lr_adam

# Define the dynamics
dynamics!(du, u, ps, t) = ude_dynamics!(du, u, ps, st, t; augmented_dynamics=nn_model)
# Define the problem
prob_nn = ODEProblem(dynamics!, IC, tspan, ps);

# Define a predictor
function predict(θ, X = IC, T = t; solver=solver, tolerance=tolerance, sense=sense, prob_nn=prob_nn)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)  # update ODE Problem with nn parameters
    Array(solve(_prob, solver, saveat = T,
                abstol=tolerance, reltol=tolerance,
                sensealg = sense
                ))
end;

t1 = now()
p_opt, st, losses, losses_regularization, Epo_degradation_BaF3, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = train_Boehm(ps, st, lr_adam, λ_reg, stepnorm_bfgs, epochs, n_mech)
runtime = (now()-t1).value/1000/60

############### Evaluate Experiment ###################
# log results
# loss curve
if create_plots
    plot_loss_trajectory(losses; path_to_store=experiment_run_path, return_plot=false)
    plot_regularization_loss_trajectory(losses_regularization; path_to_store=experiment_run_path, return_plot=false)
end

# prediction
t_plot = t_full
pred = predict(p_opt, IC, t_plot)
pred_obs = observable_mapping(pred, p_opt, st; augmented_dynamics=nn_model)

if create_plots
    # plot observables
    plot_observed_boehm(t_plot, pred_obs, t_full, y_obs_full, t, y_obs, experiment_run_path, p_opt)
    # plot hidden states
    plot_hidden_boehm(t_plot, pred, t_full, y_hidden_full, experiment_run_path)
end

# store model predictions for observables and hidden states
open(joinpath(experiment_run_path, "predictions.csv"), "w") do io
    header = ["t" "pSTAT5A_rel" "pSTAT5B_rel" "rSTAT5A_rel" "STAT5A" "STAT5B" "pApB" "pApA" "pBpB" "nucpApA" "nucpApB" "nucpBpB"]
    writedlm(io, header, ",")
    writedlm(io, [t_plot transpose(pred_obs) transpose(pred)], ",")
end
# store training results over epochs
open(joinpath(experiment_run_path, "training_curves.csv"), "w") do io
    header = ["epoch" "loss" "regularization_loss" "Epo_degradation_BaF3" "k_exp_hetero" "k_exp_homo" "k_imp_hetero" "k_imp_homo" "k_phos"]
    writedlm(io, header, ",")
    if mechanistic_setting == "boehm_missing_interaction"
        writedlm(io, [1:length(losses) losses losses_regularization Epo_degradation_BaF3 k_exp_hetero k_exp_homo repeat([missing], length(losses)) k_imp_homo k_phos], ",")
    else
        writedlm(io, [1:length(losses) losses losses_regularization Epo_degradation_BaF3 k_exp_hetero k_exp_homo k_imp_hetero k_imp_homo k_phos], ",")
    end
end

# store final values of all mechanistic parameters to csv
pars = vcat(parameter_names, noise_parameter_names)
p_mech = convert(Vector{Union{Missing,Float64}}, p_opt[1:length(pars)])
if mechanistic_setting == "boehm_missing_interaction"
    p_mech[4] = missing
end
store_parameters(experiment_run_path, pars, p_mech)
# store all nn parameters
serialize(joinpath(experiment_run_path, "nn_parameter"), p_opt)
# deserialize(joinpath(experiment_run_path, "nn_parameter"))

# add current hp settings and results to results.csv
pred = predict(p_opt, IC, t)
pred_obs = observable_mapping(pred, p_opt, st; augmented_dynamics=nn_model)

hidden_MSE = MSE(y_hidden, pred)
hidden_nMSE = nMSE(y_hidden, pred)
obs_MSE = MSE(y_obs, pred_obs)
obs_nMSE = nMSE(y_obs, pred_obs)

NegLL = nll(p_opt)

open(joinpath(experiment_series_path, "summary.csv"), "a") do io
    loss = losses[end]
    writedlm(io, [problem_name mechanistic_setting dataset sampling_strategy par_row array_nr epochs[1] epochs[2] lr_adam stepnorm_bfgs λ_reg act_fct_name hidden_layers hidden_neurons tolerance mean(hidden_MSE) mean(hidden_nMSE) mean(obs_MSE) mean(obs_nMSE) runtime loss NegLL], ",")    
end

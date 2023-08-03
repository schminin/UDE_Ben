############# Import Packages ###################
using CSV, DataFrames
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

const test_setup = true  # if used on the cluster this has to be set to false
const create_plots = true

const experiment_name = "27_07_23"

const transform = "log";
const param_range = (1e-5* (1-1e-6), 100000.0 * (1+1e-6));

const problem_name = "three_species_lotka_volterra"
exp_sampling_strategy = ("no_sampling", )
exp_mechanistic_setting = ("lv_missing_dynamics", )

const solver = KenCarp4()
const sense = ForwardDiffSensitivity()
tolerance = (1e-12, )

exp_λ_reg = (1e4, 1e3, 1e2, 1.0, 1e-2, 1e-3, )
epochs = (500, 300) # (epochs_adam, epochs_bfgs)
exp_lr_adam = (1e-4, 1e-3, 1e-2, 1e-1) # lr_bfgs = 0.1*lr_adam
exp_hidden_layers = 5#(1, 2, 3, )
exp_hidden_neurons = 3#(4, 8, 16)
exp_act_fct = ("tanh", ) # identity
exp_tolerance = (1e-12, )
exp_par_setting = (1, ) # define what rows of the startpoints.csv file to try out
exp_dataset = ("lotka_volterra_datapoints_40_noise_5.csv", )

experiments = collect(Iterators.product(exp_mechanistic_setting, exp_sampling_strategy,exp_dataset, exp_λ_reg, exp_lr_adam, 
    exp_hidden_layers, exp_hidden_neurons, exp_act_fct, exp_tolerance, exp_par_setting));

if test_setup
    array_nr = 1
else 
    array_nr = parse(Int, ARGS[1])
end

mechanistic_setting, sampling_strategy, dataset, λ_reg, lr_adam, hidden_layers, hidden_neurons, act_fct_name, tolerance, par_row = experiments[array_nr]
exp_specifics = array_nr

noise = parse(Int,chop(dataset, head = 35, tail = 4))


############# Prepare Experiment #######################
# Load functinalities
if test_setup
    epochs = (50, 20)
    include("$(problem_name)/model/create_directories_lv.jl")
    include("$(problem_name)/model/utils_lv.jl")
    include("$(problem_name)/reference.jl")
    include("$(problem_name)/model/$(mechanistic_setting).jl")
    include("$(problem_name)/model/nn_lv.jl")
else
    include("$(problem_name)/model/create_directories_lv.jl")
    include("$(problem_name)/model/utils.jl")
    include("$(problem_name)/model/$(mechanistic_setting).jl")
    include("$(problem_name)/model/nn_lv.jl")
    include("$(problem_name)/reference.jl")
end

#data_set="reference"

# Define paths
experiment_series_path, experiment_run_path, data_path, parameter_path = create_paths(problem_name, experiment_name, sampling_strategy, "$(par_row)", mechanistic_setting, dataset, "$(array_nr)")


IC, tspan, t, y_obs, t_full, y_obs_full, p_true, p_ph = load_data(data_path, problem_name, noise)

# create model 
nn_model, ps, st = create_model(act_fct_name, hidden_layers, hidden_neurons, p_ph, n_out)


dynamics!(du, u, ps, t) = ude_dynamics!(du, u, ps, st, t; nn_model)

# Define the problem
prob_nn = ODEProblem(dynamics!, IC, tspan, ps);

#Training
stepnorm_bfgs = 0.1*lr_adam

# Define a predictor
function predict(θ, X = IC, T = t; solver=solver, tolerance=tolerance, sense=sense, prob_nn=prob_nn)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)  # update ODE Problem with nn parameters
    Array(solve(_prob, solver, saveat = T,
                abstol=tolerance, reltol=tolerance,
                sensealg = sense
                ))
end;
t1 = now()
p_opt, st, losses, losses_regularization, r1, a1_1, a1_2, a1_3, r2, a2_1, a2_2, a2_3, r3, a3_1, a3_2, a3_3 = train_lv(ps, st, lr_adam, λ_reg, stepnorm_bfgs, epochs, 12)
runtime = (now()-t1).value/1000/60
t_plot = t_full
pred = predict(p_opt, IC, t)

#prediction plot
#plt = plot(t, pred[1,:], label="u1", color=1)
#plot!(t, pred[2,:], label="u2", color=2)
#plot!(t, pred[3,:], label="u3", xlabel="time", color=3)


############### Evaluate Experiment ###################
# log results
# loss curve
if create_plots
    plot_loss_trajectory(losses; path_to_store=experiment_run_path, return_plot=false)
    plot_regularization_loss_trajectory(losses_regularization; path_to_store=experiment_run_path, return_plot=false)
    plot_observed_lv(t, pred, t_full, y_obs_full, t, y_obs, experiment_run_path, p_opt )
end

#store training results over epochs
open(joinpath(experiment_run_path, "training_curves.csv"), "w") do io
    header = ["epoch" "loss" "regularization_loss" "r1" "a1_1" "a1_2" "a1_3" "r2" "a2_1" "a2_2" "a2_3" "r3" "a3_1" "a3_2" "a3_3"]
    writedlm(io, header, ",")
    if mechanistic_setting == "lv_missing_dynamics"
        writedlm(io, [1:length(losses) losses losses_regularization repeat([missing], length(losses)) a1_1 a1_2 a1_3 repeat([missing], length(losses)) a2_1 a2_2 a2_3 repeat([missing], length(losses)) a3_1 a3_2 a3_3], ",")
    else
        writedlm(io, [1:length(losses) losses losses_regularization r1 a1_1 a1_2 a1_3 r2 a2_1 a2_2 a2_3 r3 a3_1 a3_2 a3_3], ",")
    end
end
# store final values of all mechanistic parameters to csv
pars = vcat(parameter_names)
p_mech = convert(Vector{Union{Missing,Float64}}, p_opt[1:length(pars)])

#store parameters
store_parameters(experiment_run_path, pars, p_mech)


# store model predictions
open(joinpath(experiment_run_path, "predictions.csv"), "w") do io
    header = ["t" "u1" "u2" "u3"]
    writedlm(io, header, ",")
    writedlm(io, [t pred[1,:] pred[2,:] pred[3,:]], ",")
end

#hidden_MSE = MSE(y_hidden, pred)
#hidden_nMSE = nMSE(y_hidden, pred)
obs_MSE = MSE(y_obs, pred)
obs_nMSE = nMSE(y_obs, pred)

NegLL = nll(p_opt)

open(joinpath(experiment_series_path, "summary.csv"), "a") do io
    header = ["Problem name" "mechanistic setting" "Dataset" "Sampling strategy" "par row" "Array ID" "Epochs ADAM" "Epochs BFGS" "Learning rate ADAM" "Stepnorm BFGS" "λ_reg" "Activation function" "Hidden layers" "Hidden neurons" "Tolerance" " Mean MSE" "Mean nMSE" "Runtime" "Loss" "NegLL"]
    writedlm(io, header, ",")
    loss = losses[end]
    writedlm(io, [problem_name mechanistic_setting dataset sampling_strategy par_row array_nr epochs[1] epochs[2] lr_adam stepnorm_bfgs λ_reg act_fct_name hidden_layers hidden_neurons tolerance mean(obs_MSE) mean(obs_nMSE) runtime loss NegLL], ",")    
end

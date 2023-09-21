"""Evaluate single optimization experiments."""

using Lux
using ComponentArrays, SciMLSensitivity, NNlib, Optimisers, OrdinaryDiffEq, Random,
      Statistics, OneHotArrays
using Plots, Measures
using Random
using Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, DifferentialEquations
using Serialization
using ChainRulesCore
using Zygote
using JLD, JLD2
include("$(problem_name)/create_directories_lv.jl")
include("$(problem_name)/utils_lv.jl")

# set paths
problem_name = "three_species_lotka_volterra"
experiment_name = "01_09_23"
transform = "log";
experiment_output_path = "$problem_name/experiments/$experiment_name" # "evaluation/$problem_name/$experiment_name" # 

# iterate over all experiments 
array_nr = 3

# load all experiment settings
hp = load(joinpath(pwd(), experiment_output_path,"hp_settings.jld"))
#hp["lr_adam"] = (1e-3, )
experiments = collect(Iterators.product(hp["mechanistic_setting"], hp["sampling_strategy"], hp["dataset"], hp["early_stopping"],hp["one_observable"], 
    hp["λ_reg"], hp["lr_adam"], hp["hidden_layer"], hp["hidden_neurons"], hp["act_fct"], hp["tolerance"], hp["par_setting"]));

mechanistic_setting, sampling_strategy, dataset, early_stopping, one_obserbale, λ_reg, lr_adam, hidden_layers, hidden_neurons, act_fct_name, tolerance, par_row = experiments[array_nr]
experiment_series_path, experiment_run_path, data_path, parameter_path = create_paths(problem_name, experiment_name, sampling_strategy, "$(par_row)", mechanistic_setting, dataset, "$(array_nr)")

#transform = "tanh_bounds";
include("$(problem_name)/reference.jl")
include("$(problem_name)/model/$(mechanistic_setting).jl")
include("$(problem_name)/model/nn_lv.jl")

IC, tspan, t, y_obs, t_full, y_obs_full, p_true, p_ph = load_data(data_path, problem_name)

# load parameters
parameters = DataFrame(CSV.File(joinpath(experiment_run_path ,"mechanistic_parameters.csv")))
param_optd = Matrix(parameters)
param_optd[1,2]="0"
param_optd[5,2]="0"
param_optd[9,2]="0"

parameter_optd=parse.(Float64,param_optd[:,2])
# define nn
nn_model, ps, st = create_model(act_fct_name, hidden_layers, hidden_neurons, p_ph, n_out)

# Define the dynamics
dynamics!(du, u, ps, t) = ude_dynamics!(du, u, ps, st, t; nn_model)
# Define the problem
ude_prob = ODEProblem(dynamics!, IC, tspan, ps);

# Define a predictor
function predict(θ, X = IC, T = t; tolerance=tolerance, prob=ude_prob)
    _prob = remake(prob, u0 = X, tspan = (T[1], T[end]), p = θ)  # update ODE Problem with nn parameters
    Array(solve(_prob, saveat = T, verbose = false,
                save_everystep=false
                ))
end;

predict(parameter_optd, IC, t_train)

# evaluation


using DelimitedFiles
using ComponentArrays
using CSV, DataFrames


"""experiment_name, problem_name, mechanistic_setting, sampling_strategy, dataset
create all important directories necessary for training the UDE 
    #### Arguments
    - problem_name:         e.g. "Boehm"
    - experiment_name:      e.g. "HP_tuning"
    - sampling_strategy:    e.g. "optimization_endpoints" 
    - par_setting:          e.g. row of startpoints.csv
    - mechanistic_setting:  e.g. "boehm_fully_known", one of mechanistic_settings in reference.jl
    - dataset:              e.g. "original", "synthetic_datapoints_32_noise_0", etc.
    - experiment_run:       e.g. index indicating hyperparameter settings / array id
"""
function create_base_directories(problem_name::String, experiment_name::String, sampling_strategy::String, par_setting::String, mechanistic_setting::String, dataset::String, experiment_run::String)
    # create path to store results of experiments
    experiment_series_path = joinpath(pwd(), problem_name, "step_3", experiment_name)
    if !isdir(experiment_series_path)
        mkdir(experiment_series_path)
    end

    exp_mechanistic_data_path = joinpath(experiment_series_path, "$(mechanistic_setting)_$(dataset)")
    if !isdir(exp_mechanistic_data_path)
        mkdir(exp_mechanistic_data_path)
    end

    exp_sampling_path = joinpath(exp_mechanistic_data_path, sampling_strategy)
    if !isdir(exp_sampling_path)
        mkdir(exp_sampling_path)
    end

    exp_parameter_path = joinpath(exp_sampling_path, "par_$(par_setting)")
    if !isdir(exp_parameter_path)
        mkdir(exp_parameter_path)
    end

    # path to a specific hyperparameter setting of a experiment series
    exp_run_path = joinpath(exp_parameter_path, experiment_run)
    if !isdir(exp_run_path)
        mkdir(exp_run_path)
    end

    data_path = joinpath(pwd(), problem_name, "data", "$(lowercase(problem_name))_$(dataset).csv")
    parameter_path = joinpath(pwd(), problem_name, "step_2", sampling_strategy, "$(mechanistic_setting)_$(dataset)")

    return (experiment_series_path, exp_run_path, data_path, parameter_path)
end


function load_data(data_path::String, parameter_path::String, problem_name::String, parameter_row::Int)
    # true data used for plotting
    full_data = readdlm(joinpath(pwd(), problem_name, "data/boehm_reference.csv"), ','; header=true)[1]'
    t_full = full_data[1,:]
    y_obs_full = full_data[2:4, :]
    y_hidden_full = full_data[5:end, :]

    parameters, header = readdlm(joinpath(parameter_path, "startpoints.csv"), ','; header=true)

    if problem_name == "Boehm"
        p_true = (; Epo_degradation_BaF3 = 0.026982514033029,
            k_exp_hetero = 1.00067973851508e-5,
            k_exp_homo = 0.006170228086381,
            k_imp_hetero= 0.0163679184468,
            k_imp_homo= 97749.3794024716,
            k_phos= 15766.5070195731);
        p_true = ComponentVector{Float32}(p_true)
        
        if occursin("boehm_missing_interaction", parameter_path)
            p_ph = (; Epo_degradation_BaF3 = transform_par(10^(parameters[parameter_row,1])),
                k_exp_hetero = transform_par(10^(parameters[parameter_row,2])),
                k_exp_homo = transform_par(10^(parameters[parameter_row,3])),
                k_imp_hetero = 0.0,
                k_imp_homo = transform_par(10^(parameters[parameter_row,4])),
                k_phos = transform_par(10^(parameters[parameter_row,5])),
                n_pSTAT5A_rel = parameters[parameter_row,6], # log space
                n_pSTAT5B_rel = parameters[parameter_row,7], # log space
                n_rSTAT5A_rel = parameters[parameter_row,8]) # log space
        elseif occursin("boehm_missing_observable", parameter_path)
            p_ph = (; Epo_degradation_BaF3 = transform_par(10^(parameters[parameter_row,1])),
                k_exp_hetero = transform_par(10^(parameters[parameter_row,2])),
                k_exp_homo = transform_par(10^(parameters[parameter_row,3])),
                k_imp_hetero = transform_par(10^(parameters[parameter_row,4])),
                k_imp_homo = transform_par(10^(parameters[parameter_row,5])),
                k_phos = transform_par(10^(parameters[parameter_row,6])),
                n_pSTAT5A_rel = 0.0, # log space
                n_pSTAT5B_rel = parameters[parameter_row,7], # log space
                n_rSTAT5A_rel = parameters[parameter_row,8]) # 
        else # fully_known or boehm_missing_state
            p_ph = (; Epo_degradation_BaF3 = transform_par(10^(parameters[parameter_row,1])),
                k_exp_hetero = transform_par(10^(parameters[parameter_row,2])),
                k_exp_homo = transform_par(10^(parameters[parameter_row,3])),
                k_imp_hetero = transform_par(10^(parameters[parameter_row,4])),
                k_imp_homo = transform_par(10^(parameters[parameter_row,5])),
                k_phos = transform_par(10^(parameters[parameter_row,6])),
                n_pSTAT5A_rel = parameters[parameter_row,7], # log space
                n_pSTAT5B_rel = parameters[parameter_row,8], # log space
                n_rSTAT5A_rel = parameters[parameter_row,9]) # log space
        end
    end
    
    # data used for training
    df = CSV.read(data_path,  DataFrame, delim="\t")
    df = unstack(df, :time, :observableId, :measurement)
    t = df[!, "time"]
    y_obs = identity.(transpose(Array(df[!, 2:end])))

    # data used for training
    idx = findall(in(t), t_full)
    y_hidden = y_hidden_full[:, idx];

    IC = y_hidden_full[:,1];
    tspan = (t_full[1], t_full[end])
    return (IC, tspan, t, y_obs, y_hidden, t_full, y_obs_full, y_hidden_full, p_true, p_ph)
end
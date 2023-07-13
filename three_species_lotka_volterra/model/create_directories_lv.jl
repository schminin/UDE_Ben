using DelimitedFiles
using ComponentArrays
using CSV, DataFrames
using Random

include("../reference.jl")

function create_paths(problem_name::String, experiment_name::String, sampling_strategy::String, par_setting::String, mechanistic_setting::String, dataset::String, experiment_run::String)
    # create path to store results of experiments
    experiment_series_path = joinpath(pwd(), problem_name, "experiments", experiment_name)
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

    data_path = joinpath(pwd(), problem_name, "data", "$(dataset)")
    parameter_path = joinpath(pwd(), problem_name, "experiments", sampling_strategy, "$(mechanistic_setting)_$(dataset)")

    return (experiment_series_path, exp_run_path, data_path, parameter_path)
end

function load_data(data_path::String,problem_name::String)

    # true data used for plotting
    full_data = readdlm(joinpath(pwd(), problem_name, "data/lotka_volterra_reference.csv"), ','; header=true)[1]'
    t_full = full_data[1,:]
    y_obs_full = full_data[2:4, :]
    #y_hidden_full = full_data[5:end, :]

    #erstmal manuell
    p_true =(; r1 = 3.0,
                a1_1 = 2.8, 
                a1_2 = 6.0,
                a1_3 = 2.0,
                r2 = 1.1,
                a2_1 = 1.8,
                a2_2 = 0.5,
                a2_3 = 2.8,
                r3 = 4.0,
                a3_1 = 3.0,
                a3_2 = 6.0,
                a3_3 =0.0)
    p_true = ComponentVector{Float32}(p_true)

    rng = Random.default_rng()
    Random.seed!(rng, 1)

    random_vec = rand(12)


    p_ph =(; r1 = transform_par(random_vec[1]),
            a1_1 = transform_par(random_vec[2]), 
            a1_2 = transform_par(random_vec[3]),
            a1_3 = transform_par(random_vec[4]),
            r2 = transform_par(random_vec[5]),
            a2_1 = transform_par(random_vec[6]),
            a2_2 = transform_par(random_vec[7]),
            a2_3 = transform_par(random_vec[8]),
            r3 = transform_par(random_vec[9]),
            a3_1 = transform_par(random_vec[10]),
            a3_2 = transform_par(random_vec[11]),
            a3_3 = transform_par(random_vec[12]),
            var1 = 0,
            var2 = 0,
            var3 = 0)

    #data used for training
    df =CSV.read(data_path, DataFrame, delim = ",")
    t = df[!, "t"]
    y_obs = identity.((Array(df[!, 2:end])))

    # data used for training
    idx = findall(in(t), t_full)
    tspan = (t[1], t[end])

    return (IC, tspan, t, y_obs, t_full, y_obs_full, p_true, p_ph)
end
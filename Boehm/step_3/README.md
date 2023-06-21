In step 3, the UDE is trained using the mechanistic parameters identified in step 2 as starting points.

#### The file structure is as follows:
| File Name      | Description |
| ----------- | ----------- |
| reference.jl      | overview of different variable names used in the project       |
| utils.jl   | utils used to define the UDE model, as well as for training the model, <br> plotting figures related to step_3 and evaluation of the training process        |
| create_directories.jl | functions to define paths, create directories and load data |
| model/* | definition of the neural network component of the UDE and each mechanistic setting <br> (containing formulations for the UDE and the observable mapping)  |
| experiment.jl | definition of experimental settings and implementation of the experiment |
| job_file.mpi | mpi file used to run a job on the cluster |
| log/* | log directory containing *.out files of experiment runs on the cluster |


#### To run a new experiment:
1. Change the experimental settings in experiment.jl. 
    * This is equivalent to setting a grid of settings to explore.
    * If you use only one setting of a hyperparameter, ensure to use "," afterwards. I.e. write ```(10, )``` instead of ```(10)```.
2. Each setting is equivalent to one SLURM array id. Hence, depending on the number of settings to explore (which equals ```length(experiments)```) the array number in job_file.mpi has to be updated.
3. Results:
    * The log files are stored in the log directory as described above. 
    * Everything else is stored in the ```experiment_name``` directory.
    * The summary.csv file summarises the results of the training and gives the mapping ```array_nr``` <-> experiment setting.
    * While the names of the folders of the experiment folder structure the results according to mechanistic setting, data and sampling strategy the deepest folder structure is named according to array_ids.

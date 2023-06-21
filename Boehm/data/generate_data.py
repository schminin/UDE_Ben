import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

import benchmark_models_petab
import petab
from petab.C import (
    OBSERVABLE_ID,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
    MEASUREMENT,
    TIME,
    OBSERVABLE_PARAMETERS,
    NOISE_PARAMETERS,
    DATASET_ID,
)
import petab.visualize
import amici
import amici.petab_simulate
from amici.petab_import import import_petab_problem

data_dir = Path(__file__).resolve().parent

petab_problem = benchmark_models_petab.get_problem("Boehm_JProteomeRes2014")

# original data
petab.write_measurement_df(
    df=petab_problem.measurement_df, filename=str(data_dir / "boehm_original.csv")
)
# Visualization
populations = {
    "pSTAT5A_rel": {
        "observable_id": "pSTAT5A_rel",
        "plot": {
            "color": "black",
        },
    },
    "pSTAT5B_rel": {
        "observable_id": "pSTAT5B_rel",
        "plot": {
            "color": "red",
        },
    },
    "rSTAT5A_rel": {
        "observable_id": "rSTAT5A_rel",
        "plot": {
            "color": "green",
        },
    },
}
fig, ax = plt.subplots()
df = petab_problem.measurement_df
for population, population_settings in populations.items():
    index = df.observableId == population_settings["observable_id"]
    ax.scatter(
        df.time.loc[index],
        df.measurement.loc[index],
        s=4,
        **population_settings["plot"],
        label=population,
    )
ax.legend()
fig.set_size_inches((6, 5))
plt.savefig(str(data_dir / "boehm_original.png"))

# generate synthetic data

t_synthetic = [
    0,
    1,
    2,
    3,
    4,
    5,
    7.5,
    10,
    12.5,
    15,
    17.5,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
]
tn = len(t_synthetic)

# set new timepoints in measurement table
petab_problem.measurement_df = pd.DataFrame(
    {
        OBSERVABLE_ID: (
            ["pSTAT5A_rel"] * tn + ["pSTAT5B_rel"] * tn + ["rSTAT5A_rel"] * tn
        ),
        # PREEQUILIBRATION_CONDITION_ID: [np.nan] * (tn * 3),
        SIMULATION_CONDITION_ID: ["model1_data1"] * (tn * 3),
        MEASUREMENT: [np.nan] * (tn * 3),  # to be simulated
        TIME: np.hstack([t_synthetic, t_synthetic, t_synthetic]),
        # OBSERVABLE_PARAMETERS: [np.nan] * (tn * 3),
        NOISE_PARAMETERS: (
            ["sd_pSTAT5A_rel"] * tn + ["sd_pSTAT5B_rel"] * tn + ["sd_rSTAT5A_rel"] * tn
        ),
        DATASET_ID: (
            ["model1_data1_pSTAT5A_rel"] * tn
            + ["model1_data1_pSTAT5B_rel"] * tn
            + ["model1_data1_rSTAT5A_rel"] * tn
        ),
    }
)

# simulate
rng = np.random.default_rng(seed=0)
simulator = amici.petab_simulate.PetabSimulator(petab_problem=petab_problem)
simulator.rng = rng
sim_df = simulator.simulate()

obs_means = df.groupby(by="observableId").measurement.mean()
obs_means = obs_means.loc[obs_means.index.repeat(tn)]

for noise_level in [0, 0.05, 0.2]:
    noise_percent = int(noise_level * 100)

    noisy_df = deepcopy(sim_df)
    noisy_df[MEASUREMENT] = rng.normal(
        loc=noisy_df[MEASUREMENT],
        scale=obs_means * noise_level,
    )
    petab_problem.measurement_df = noisy_df
    petab.write_measurement_df(
        df=petab_problem.measurement_df,
        filename=str(data_dir / f"boehm_synthetic_datapoints_{tn}_noise_{noise_percent}.csv"),
    )
    fig, ax = plt.subplots()
    df = noisy_df
    for population, population_settings in populations.items():
        index = df.observableId == population_settings["observable_id"]
        ax.scatter(
            df.time.loc[index],
            df.measurement.loc[index],
            s=4,
            **population_settings["plot"],
            label=population,
        )
    ax.legend()
    fig.set_size_inches((6, 5))
    plt.savefig(str(data_dir / f"boehm_synthetic_datapoints_{tn}_noise_{noise_percent}.png"))

# reference for plotting
petab_problem = benchmark_models_petab.get_problem("Boehm_JProteomeRes2014")
amici_model = import_petab_problem(petab_problem)
# set narrow timepoints
t_ref = np.around(np.arange(0, 240.1, 0.1), decimals=1)
amici_model.setTimepoints(t_ref)
# simulate
solver = amici_model.getSolver()
rdata = amici.runAmiciSimulation(amici_model, solver)
# observables
observables_ids = petab_problem.observable_df.index.values
ref_df = pd.DataFrame({"time": t_ref})
for i, obs_id in enumerate((observables_ids)):
    ref_df[obs_id] = rdata["y"][:, i]
# states
states_ids = [
    "STAT5A",
    "STAT5B",
    "pApB",
    "pApA",
    "pBpB",
    "nucpApA",
    "nucpApB",
    "nucpBpB",
]
for i, state_id in enumerate((states_ids)):
    ref_df[state_id] = rdata["x"][:, i]
ref_df.to_csv(data_dir / "boehm_reference.csv", index=False, header=True)

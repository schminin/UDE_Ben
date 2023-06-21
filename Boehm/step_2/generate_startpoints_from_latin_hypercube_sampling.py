"""Strategy: Sample and evaluate objective funtion."""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import shiftedColorMap

import amici
import pypesto
import pypesto.petab

from Boehm.sym_model_importer import experiments
from utils import get_boehm_parameters

boehm_dir = Path(__file__).resolve().parents[1]
n_startpoints = 20

for model_sym_id in experiments.keys():
    petab_problem = experiments[model_sym_id]["petab problem"]
    pypesto_importer = experiments[model_sym_id]["pypesto importer"]
    amici_model_dir = experiments[model_sym_id]["amici model dir"]
    # amici model is compiled with objective:
    pypesto_objective = pypesto_importer.create_objective()
    amici_model = pypesto_objective.amici_model
    amici_solver = pypesto_objective.amici_solver
    amici_solver.setMaxSteps(int(1e5))
    amici_solver.setAbsoluteTolerance(1e-8)
    amici_solver.setRelativeTolerance(1e-6)
    amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
    pypesto_problem = pypesto_importer.create_problem(objective=pypesto_objective)

    output_dir = boehm_dir / "step_2" / "latin_hypercube_sampling" / model_sym_id
    os.makedirs(output_dir, exist_ok=True)

    startpoint_method = pypesto.startpoint.LatinHypercubeStartpoints(
        use_guesses=False,
        check_fval=True,
        check_grad=True,
    )
    startpoints = startpoint_method(
        n_starts=1000,
        problem=pypesto_problem,
    )

    def set_fix_values(x):
        for i, val in zip(
            pypesto_problem.x_fixed_indices, pypesto_problem.x_fixed_vals
        ):
            x[i] = val
        return x

    fvals = [
        pypesto_objective.get_fval(
            set_fix_values(pypesto_problem.get_full_vector(startpoint))
        )
        for startpoint in startpoints
    ]

    parameters = get_boehm_parameters(model_sym_id)

    df = pd.DataFrame(startpoints, columns=parameters)
    df["objective_function_value"] = fvals
    df.sort_values(by="objective_function_value", inplace=True, ignore_index=True)
    # subset
    df_20 = df.iloc[:n_startpoints, :]
    df_20.to_csv(output_dir / "startpoints.csv", index=False)

    # visualizations

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(fvals[:100])
    ax[1].plot(fvals)
    ax[1].set_yscale("log")
    fig.suptitle("Objective function values")
    fig.savefig(output_dir / "function_values.png")

    cmap = shiftedColorMap(
        cm.Blues_r, start=0, midpoint=0.4, stop=0.8, name="visible_blues_r"
    )

    fig, ax = plt.subplots(1, 1)
    df_20[parameters].T.plot(legend=True, colormap=cmap, ax=ax, rot=25)
    ax.legend(
        labels=[f"{i}." for i in range(1, n_startpoints + 1)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.set_ylim(-5, 5)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.set_title("Best parameter vectors from latin hypercube sampling")
    fig.tight_layout()
    fig.savefig(output_dir / "startpoints.png")

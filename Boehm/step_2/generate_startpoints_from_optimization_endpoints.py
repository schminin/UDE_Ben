import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import shiftedColorMap

import amici
import pypesto
import pypesto.petab
import pypesto.ensemble as ensemble

from utils import get_boehm_parameters
from Boehm.sym_model_importer import experiments

boehm_dir = Path(__file__).resolve().parents[1]


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

    output_dir = boehm_dir / "step_2" / "optimization_endpoints" / model_sym_id
    os.makedirs(output_dir, exist_ok=True)

    n_startpoints = 15

    # load results
    results_path = boehm_dir / "step_1" / model_sym_id / "optimization_result.hdf5"
    result = pypesto.store.read_result(
        results_path, optimize=True, profile=False, sample=False
    )

    ensemble_model = ensemble.Ensemble.from_optimization_endpoints(
        result=result,
        max_size=n_startpoints,
    )

    startpoints = ensemble_model.x_vectors[pypesto_problem.x_free_indices, :].T
    df = pd.DataFrame(startpoints, columns=ensemble_model.x_names)
    df["vector_tag"] = ensemble_model.vector_tags
    df.to_csv(output_dir / "startpoints.csv", index=False)

    # show parameter vectors
    parameters = get_boehm_parameters(model_sym_id)
    cmap = shiftedColorMap(
        cm.Blues_r, start=0, midpoint=0.4, stop=0.8, name="visible_blues_r"
    )

    fig, ax = plt.subplots(1, 1)
    df[parameters].T.plot(legend=True, colormap=cmap, ax=ax, rot=25)
    ax.legend(
        labels=[f"{i}." for i in range(1, n_startpoints + 1)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.set_ylim(-5, 5)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.set_title("Best parameter vectors from pypesto optimization endpoints")
    fig.tight_layout()
    fig.savefig(output_dir / "startpoints.png")

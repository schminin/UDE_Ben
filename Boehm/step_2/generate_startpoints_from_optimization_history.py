"""Try the ensemble model from optimizatoin history."""

import os
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

import amici
import pypesto
import pypesto.petab
import pypesto.ensemble as ensemble

boehm_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(boehm_dir))

from sym_model_importer import experiments

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

    output_dir = boehm_dir / "step_2" / "optimization_history" / model_sym_id
    os.makedirs(output_dir, exist_ok=True)

    n_startpoints = 15

    # load results
    results_path = boehm_dir / "step_1" / model_sym_id / "optimization_result.hdf5"
    result = pypesto.store.read_result(
        results_path,
        optimize=True,
        profile=False,
        sample=False,
    )
    # load history
    history_path = boehm_dir / "step_1" / model_sym_id / "optimization_history.hdf5"
    for res in result.optimize_result.list:
        res.history = pypesto.Hdf5History.load(res.id, history_path)

    ensemble_model = ensemble.Ensemble.from_optimization_history(
        result=result,
        max_size=n_startpoints,
        max_per_start=5,
    )

    startpoints = ensemble_model.x_vectors.T
    df = pd.DataFrame(startpoints, columns=ensemble_model.x_names)
    df["vector_tag"] = ensemble_model.vector_tags
    df.to_csv(output_dir / "startpoints.csv", index=False)

    # show parameter vectors
    mechanistic_parameters = [
        "Epo_degradation_BaF3",
        "k_exp_hetero",
        "k_exp_homo",
        "k_imp_hetero",
        "k_imp_homo",
        "k_phos",
    ]
    if "missing_interaction" in model_sym_id:
        del mechanistic_parameters[3]
    fig, ax = plt.subplots(1, 1)
    df[mechanistic_parameters].T.plot(legend=True, colormap="Blues_r", ax=ax, rot=25)
    ax.legend([f"{i}." for i in range(1, n_startpoints + 1)])
    ax.set_ylim(-5, 5)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.set_title("Best parameter vectors from pypesto optimization endpoints")
    fig.tight_layout()
    fig.savefig(output_dir / "startpoints.png")

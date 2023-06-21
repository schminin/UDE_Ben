"""Calibrate the models with partially missing dynamics."""

import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import amici
import fides

import matplotlib.pyplot as plt

from Boehm.sym_model_importer import experiments

from pathlib import Path
boehm_dir = results_path = Path(__file__).resolve().parents[1]

for experiment_id in experiments.keys():
    petab_problem = experiments[experiment_id]["petab problem"]
    pypesto_importer = experiments[experiment_id]["pypesto importer"]
    amici_model_dir = experiments[experiment_id]["amici model dir"]

    # amici model is compiled with objective:
    pypesto_objective = pypesto_importer.create_objective()
    amici_model = pypesto_objective.amici_model
    amici_solver = pypesto_objective.amici_solver
    amici_solver.setMaxSteps(int(1e5))
    amici_solver.setAbsoluteTolerance(1e-8)
    amici_solver.setRelativeTolerance(1e-6)
    amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
    pypesto_problem = pypesto_importer.create_problem(objective=pypesto_objective)

    # Optimization settings
    n_starts = 100
    optimizer = optimize.FidesOptimizer(
        verbose=0,
        hessian_update=fides.BFGS(),
        options={'maxiter': 10000},
    )
    engine = pypesto.engine.MultiProcessEngine(6)
    results_dir = boehm_dir / "step_1" / experiment_id
    history_options = pypesto.HistoryOptions(
        trace_record=True, 
        trace_save_iter=10, 
        storage_file=str(results_dir / "optimization_history.hdf5"),
    )
    result = optimize.minimize(
        problem=pypesto_problem,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=engine,
        history_options=history_options,
    )
    pypesto.store.write_result(
        result=result, 
        filename=results_dir / "optimization_result.hdf5", 
        optimize=True,
    )
    pypesto.visualize.waterfall(result)
    plt.savefig(results_dir / "waterfall.png")

"""Convergence curve for the Poisson equation with manufactured solution
Based on: https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html"""
# pylint: disable=logging-fstring-interpolation, c-extension-no-member, import-error
import copy
import logging

import numpy as np
import pandas as pd
import ufl
from dolfinx import fem
from mpi4py import MPI
from yaml import safe_load

from poisson_vanilla import solve_poisson

# Set up custom logging
custom_logger = logging.getLogger("poisson_manufactured")
logging.basicConfig(
    level=logging.INFO,
)


if __name__ == "__main__":
    # Set up empty DataFrame to store results
    results_h = pd.DataFrame(columns=["h_refinement", "l2_error_h"])
    results_p = pd.DataFrame(columns=["p_refinement", "l2_error_p"])

    # Load configuration file
    with open("poisson_config.yml", "r", encoding="utf-8") as file:
        cfg = safe_load(file)

    # h and p refinement parameters
    h_ref_iterations = cfg["manufactured"]["h_refinement_iterations"]
    p_ref_iterations = cfg["manufactured"]["p_refinement_iterations"]

    # https://stackoverflow.com/questions/2465921/how-to-copy-a-dictionary-and-only-edit-the-copy
    # Create deep copies of the configuration for h and p refinement
    h_ref_cfg = copy.deepcopy(cfg)
    p_ref_cfg = copy.deepcopy(cfg)

    # h refinement
    for h_ref in range(h_ref_iterations):
        V, u_h = solve_poisson(h_ref_cfg)
        u_D = fem.Function(V)
        u_D.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
        error = fem.form((u_D - u_h) ** 2 * ufl.dx)
        l2_error = np.sqrt(
            MPI.COMM_WORLD.allreduce(fem.assemble_scalar(error), op=MPI.SUM)
        )
        custom_logger.info(
            f"h refinement iteration {h_ref + 1}: L2 error = {l2_error:.6e}"
        )

        h_ref_cfg["mesh"]["nx"] *= 2
        h_ref_cfg["mesh"]["ny"] *= 2

        results_h = pd.concat(
            [
                results_h,
                pd.DataFrame(
                    {"h_refinement": [f"h_ref_{h_ref + 1}"], "l2_error_h": [l2_error]}
                ),
            ],
            ignore_index=True,
        )

    # p refinement
    for p_ref in range(p_ref_iterations):
        V, u_h = solve_poisson(p_ref_cfg)
        u_D = fem.Function(V)
        u_D.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
        error = fem.form((u_D - u_h) ** 2 * ufl.dx)
        l2_error = np.sqrt(
            MPI.COMM_WORLD.allreduce(fem.assemble_scalar(error), op=MPI.SUM)
        )
        custom_logger.info(
            f"p refinement iteration {p_ref + 1}: L2 error = {l2_error:.6e}"
        )

        p_ref_cfg["mesh"]["element_order"] += 1

        results_p = pd.concat(
            [
                results_p,
                pd.DataFrame(
                    {"p_refinement": [f"p_ref_{p_ref + 1}"], "l2_error_p": [l2_error]}
                ),
            ],
            ignore_index=True,
        )

    # Save results to CSV files
    results_h.to_csv("h_refinement_results.csv", index=False)
    results_p.to_csv("p_refinement_results.csv", index=False)

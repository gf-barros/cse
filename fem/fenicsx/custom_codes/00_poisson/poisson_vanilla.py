""" This script solves the Poisson equation on a rectangular domain using FEniCSx."""
# pylint: disable=invalid-name, no-name-in-module, c-extension-no-member
import logging
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from yaml import safe_load


# Set up custom logging
custom_logger = logging.getLogger("poisson_vanilla")
logging.basicConfig(
    level=logging.INFO,
)

def solve_poisson(cfg=None):
    """Main function to solve the Poisson equation."""

    # Set up output directory
    root_dir = Path(cfg["output_file"]["root_dir"])
    file_dir = root_dir / cfg["output_file"]["dir"]
    filename = cfg["output_file"]["file_name"]

    # Mesh creation
    nx, ny = cfg["mesh"]["nx"], cfg["mesh"]["ny"]
    point_1 = np.array(cfg["mesh"]["point_1"])
    point_2 = np.array(cfg["mesh"]["point_2"])
    cell_type = cfg["mesh"]["element_type"]
    element_order = cfg["mesh"]["element_order"]

    # Update filename to include mesh parameters
    filename = filename + f"_{nx}_{ny}_{cell_type}_ord{element_order}"

    # Create mesh and function space from config file
    cell_type_dict = {
        "triangle": mesh.CellType.triangle,
        "quadrilateral": mesh.CellType.quadrilateral,
    }

    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=[point_1, point_2],
        n=[nx, ny],
        cell_type=cell_type_dict[cell_type],
    )
    V = fem.functionspace(domain, ("Lagrange", element_order))

    custom_logger.info(f"Mesh created with {nx}x{ny} cells of type {cell_type} and order {element_order}")
    custom_logger.info(f"Total number of dofs in mesh: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

    # Create boundary condition
    fdim = domain.topology.dim - 1

    # Left and right Dirichlet BCs
    left_and_right_boundaries_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    )
    left_and_right_boundaries_dofs = fem.locate_dofs_topological(
        V, fdim, left_and_right_boundaries_facets
    )
    bc_left_and_right = fem.dirichletbc(
    ScalarType(0), left_and_right_boundaries_dofs, V
    )

    # Top and bottom Dirichlet BCs
    top_and_bottom_boundaries_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    )
    top_and_bottom_boundary_functions = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    top_and_bottom_boundary_functions.interpolate(
        lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    )
    top_and_bottom_boundary_dofs = fem.locate_dofs_topological(
        V, fdim, top_and_bottom_boundaries_facets
    )
    bc_top_and_bottom = fem.dirichletbc(
        top_and_bottom_boundary_functions, top_and_bottom_boundary_dofs
    )
    bc = [bc_left_and_right, bc_top_and_bottom]

    # Variational Formulation
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = 2 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1])
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Solve the problem
    problem = LinearProblem(
        a, L, bcs=bc, petsc_options={
            **cfg["solver"],}
    )
    u_h = problem.solve()
    u_h.name = "u_h"

    # Validate solution and compute error
    custom_logger.info(f"Minimum value of u_h: {u_h.x.petsc_vec.array.min()}")
    custom_logger.info(f"Maximum value of u_h: {u_h.x.petsc_vec.array.max()}")

    # Write solution and close the xdmf file
    if element_order == 1:
        filename = filename + ".xdmf"
        u_file = io.XDMFFile(domain.comm, file_dir / filename, "w")
        u_file.write_mesh(domain)
        u_file.write_function(u_h, 0.0)
        u_file.close()
    else:
        filename = filename + ".bp"
        u_file = io.VTXWriter(domain.comm, file_dir / filename, u_h, engine="BP5")
        u_file.write(0.0)

    return V, u_h

if __name__ == "__main__":
    # Load configuration file
    with open("poisson_config.yml", "r", encoding="utf-8") as file:
        cfg = safe_load(file)

    # Solve the Poisson equation
    solve_poisson(cfg)
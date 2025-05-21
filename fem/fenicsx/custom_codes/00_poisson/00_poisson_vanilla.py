import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem
from pathlib import Path
from yaml import safe_load

def solve_poisson():

    # Load configuration
    with open("poisson_config.yml", "r") as file:
        cfg = safe_load(file)

    # Set up output directory
    ROOT_DIR = Path(cfg["output_file"]["root_dir"])
    DIR = ROOT_DIR / cfg["output_file"]["dir"]
    filename = cfg["output_file"]["file_name"] 


    # Mesh creation
    nx, ny = cfg["mesh"]["nx"], cfg["mesh"]["ny"]
    point_1 = np.array(cfg["mesh"]["point_1"])
    point_2 = np.array(cfg["mesh"]["point_2"])
    cell_type = cfg["mesh"]["element_type"]
    element_order = cfg["mesh"]["element_order"]

    # Update filename to include mesh parameters
    filename = filename + f"_{nx}_{ny}_{cell_type}_ord{element_order}"

    cell_type_dict = {
        "triangle": mesh.CellType.triangle,
        "quadrilateral": mesh.CellType.quadrilateral,
    }

    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD, 
        points=[point_1, point_2],
        n=[nx, ny],
        cell_type=cell_type_dict[cell_type])
    V = fem.functionspace(domain, ("Lagrange", element_order))
    u_h = fem.Function(V)
    u_h.name = "u_h"

    # Create boundary condition
    fdim = domain.topology.dim - 1
    left_and_right_boundaries_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
    left_and_right_boundaries_dofs = fem.locate_dofs_topological(V, fdim, left_and_right_boundaries_facets)
    bc_left_and_right = fem.dirichletbc(PETSc.ScalarType(0), left_and_right_boundaries_dofs, V)

    top_and_bottom_boundaries_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    top_and_bottom_boundary_functions = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    top_and_bottom_boundary_functions.interpolate(lambda x: np.sin(np.pi*x[0])*np.cos(np.pi*x[1]))
    top_and_bottom_boundary_dofs = fem.locate_dofs_topological(V, fdim, top_and_bottom_boundaries_facets)
    bc_top_and_bottom = fem.dirichletbc(top_and_bottom_boundary_functions, top_and_bottom_boundary_dofs)
    bc = [bc_left_and_right, bc_top_and_bottom]

    # Create output file and save mesh
    if element_order == 1:
        filename = filename + ".xdmf"
        u_file = io.XDMFFile(domain.comm, DIR / filename, "w")
        u_file.write_mesh(domain)
    # else:
    #     filename = filename + ".bp"
    #     u_file = io.VTXWriter(domain.comm, DIR / filename, u_h, engine="BP5")


    # Variational Formulation
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = 2*np.pi**2*ufl.sin(np.pi*x[0])*ufl.cos(np.pi*x[1])
    a =  ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L =  f * v * ufl.dx

    # Solve the problem
    problem = LinearProblem(a, L, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_h = problem.solve()
    u_h.name = "u_h"

    print(u_h.x.petsc_vec.array.min(), u_h.x.petsc_vec.array.max())

    # Write solution and close the xdmf file
    if element_order == 1:
        u_file.write_function(uh, 0.0)
    else:
        filename = filename + ".bp"
        u_file = io.VTXWriter(domain.comm, DIR / filename, u_h, engine="BP5")
        u_file.write(0.0)

    # u_file.close()

if __name__ == "__main__":
    solve_poisson()
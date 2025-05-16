""" Code used to process the lock-exchange simulation using FEniCSx"""

import gmsh
from mpi4py import MPI
import numpy as np
from dolfinx import mesh, io, fem, default_real_type, log
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver 
from dolfinx.fem.petsc import NonlinearProblem
import basix.ufl
import ufl
from pathlib import Path
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
psize = comm.Get_size()
print("psize = ", psize)


# =====================================================================|
#                                                                      |
#                                                                      |
#                               HEADER                                 |
#                                                                      |
#                                                                      |
# =====================================================================|

# # General flags
EXPORT_INTERVAL = 1
ROOT_DIR = Path("/mnt/c/ubuntu_interface/")
DIR = ROOT_DIR / "cse_custom_codes/navier_stokes/"

# # Time parameters
DT = 0.01
TOTAL_T = 0.1

# Domain + Mesh
HEIGHT = 2.05
WIDTH = 18.0
NX = 700
NY = 100

gmsh.initialize()

L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
res_min = r / 3
gdim = 2
fdim = gdim - 1
mesh_comm = MPI.COMM_WORLD
model_rank = 0
fluid_marker = 1
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle_list = [], [], [], []

if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle_list.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle_list, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle_list)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.optimize("Netgen")

domain, _, ft = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"

# gmsh.finalize()

# Problem setup
el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
el_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

el_mixed = basix.ufl.mixed_element([el_u, el_p])

V = fem.functionspace(domain, el_u)
Q = fem.functionspace(domain, el_p)
W = fem.functionspace(domain, el_mixed)


w, q = ufl.TestFunctions(W)
wh = fem.Function(W)
wh_n = fem.Function(W)


v_function, p_function = wh.split()
vn, pn = wh_n.split()

class InletVelocity():
    def __init__(self):
        pass

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * x[1] * (0.41 - x[1]) / (0.41**2)
        return values


# Inlet
u_inlet = fem.Function(V)
inlet_velocity = InletVelocity()
u_inlet.interpolate(inlet_velocity)
bcu_inflow = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(V, 1, ft.find(inlet_marker)))

# Walls
u_nonslip = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, 1, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, 1, ft.find(obstacle_marker)), V)

# Outlet
bcp_outlet = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcs = [bcu_inflow, bcu_walls, bcu_obstacle, bcp_outlet]

# Export mesh to disk
xdmf_solution = io.XDMFFile(domain.comm, DIR / "solution.xdmf", "w")
xdmf_solution.write_mesh(domain)
xdmf_solution.write_function(v_function, 0.0)
xdmf_solution.write_function(p_function, 0.0)


# Setup Variational Problem
dtt = fem.Constant(domain, DT)
h = ufl.CellDiameter(domain)
RE = 100.0
NU = fem.Constant(domain, 1 / RE)
F_FORCE = fem.Constant(domain, (0.0, 0.0))  # body forces

vnorm = ufl.sqrt(ufl.dot(v_function, v_function))
# Use these in F instead of separate v_function, c_function
F = (
    ufl.inner(v_function - vn, w) / dtt
    + ufl.inner(ufl.dot(v_function, ufl.nabla_grad(v_function)), w)
    + NU * ufl.inner(ufl.grad(w), ufl.grad(v_function))
    - ufl.inner(p_function, ufl.div(w))
    + ufl.inner(q, ufl.div(v_function))
    - ufl.inner(F_FORCE, w)
) * ufl.dx 

# Then the stabilization terms, replacing v_function and c_function by these splits too
v, p = ufl.split(wh)
vnorm = ufl.sqrt(ufl.dot(v, v))
h = ufl.CellDiameter(domain)
R = (
    (1.0 / dtt) * (v - vn)
    + ufl.dot(v, ufl.nabla_grad(v))
    - NU * ufl.div(ufl.grad(v))
    + ufl.grad(p)
    - F_FORCE
)

epsilon = 1e-10
tau = ((2.0 / dtt) ** 2 + (2.0 * vnorm / (h + epsilon)) ** 2 + 9.0 * (4.0 * NU / ((h + epsilon) * 2)) ** 2) ** (-0.5)

tau_lsic = (vnorm * h) / 2.0
F_lsic = tau_lsic * ufl.inner(ufl.div(v_function), ufl.div(w)) * ufl.dx
F_supg = tau * ufl.inner(R, ufl.dot(v_function, ufl.nabla_grad(w))) * ufl.dx
F_pspg = tau * ufl.inner(R, ufl.grad(q)) * ufl.dx

F += F_supg + F_pspg + F_lsic

# Pass the current solution function wh (not space) to NonlinearProblem
problem = NonlinearProblem(F, wh, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# # Set convergence criteria
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.atol = 1e-6
solver.max_it = 40
solver.report = True

# # # Configure the Krylov solver (linear solver used within Newton iterations)
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-6
opts[f"{option_prefix}ksp_atol"] = 1.0e-6
opts[f"{option_prefix}ksp_max_it"] = 500
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

ksp.setFromOptions()
log.set_log_level(log.LogLevel.INFO)


STEP = 0
T = 0.0

while T < TOTAL_T:
    if rank == 0:
        print(f"\nt = {T:10.3e}\n")

    n, converged = solver.solve(wh)
    if not converged:
        print("Solver failed to converge.")
        break
    print(f"Solver converged with {n} Newton iterations.")
    
    with wh.x.petsc_vec.localForm() as loc, wh_n.x.petsc_vec.localForm() as loc_n:
        loc_n.copy(loc)
    wh_n.x.scatter_forward()        
    T = T + DT
    STEP += 1


    if STEP % EXPORT_INTERVAL == 0:
        v_function, p_function = wh.split()
        xdmf_solution.write_function(v_function, T)
        xdmf_solution.write_function(p_function, T)
        print(f"Exporting at t = {T:10.3e}...")

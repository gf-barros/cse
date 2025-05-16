""" Code used to process the lock-exchange simulation using FEniCSx"""

from mpi4py import MPI
import numpy as np
from dolfinx import mesh, io, fem, default_real_type, log
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver 
from dolfinx.fem.petsc import NonlinearProblem
import basix.ufl
import ufl
from pathlib import Path
# from turbidity_utils import (
#     InitialConditionsNS,)

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
# SUPG = True
ROOT_DIR = Path("/mnt/c/ubuntu_interface/")
DIR = ROOT_DIR / "cse_custom_codes/turbidity/"
# ITER_SOLVER_NS = True  # True = GMRES / False = LU
# SUPS = True  # True = SUPS / False = Taylor-Hood
# LSIC = True


# # Parameters
GR = np.sqrt(5e6)  # Grashof
SC = 1.00  # Schmidt



# # Time parameters
# T = 0.0
DT = 0.01
TOTAL_T = 0.1


# Domain + Mesh
HEIGHT = 2.05
WIDTH = 18.0
NX = 700
NY = 100


# =================      FENICS PREPROCESSING     ======================
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True

# =====================================================================|
#                                                                      |
#                                                                      |
#                           PROCESSING                                 |
#                                                                      |
#                                                                      |
# =====================================================================|


domain = mesh.create_rectangle(
    MPI.COMM_WORLD, 
    [np.array([0., 0.]), np.array([WIDTH, HEIGHT])], 
    [NX, NY],
    mesh.CellType.triangle,)

# Export mesh to disk
xdmf_solution = io.XDMFFile(domain.comm, DIR / "solution.xdmf", "w")
xdmf_solution.write_mesh(domain)

# Problem setup
el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
el_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_c = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_mixed = basix.ufl.mixed_element([el_u, el_p, el_c])

W = fem.functionspace(domain, el_mixed)
vpc = ufl.TrialFunction(W)
v, p, c = ufl.split(vpc)
(w, q, qc) = ufl.TestFunctions(W)

v = ufl.as_vector([vpc[0], vpc[1]])
p = vpc[2]
c = vpc[3]

wh = fem.Function(W)
wh.x.scatter_forward()
wh_n = fem.Function(W)

import random

def y0_init(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = 0.0 + 0.2 * (0.5 - random.random())
    return values

def y1_init(x):
    values = np.zeros((1, x.shape[1]))
    x0 = x[0]
    x1 = x[1]

    # Condition 1
    mask1 = (x0 < 1.05) & (x1 < 2.00)
    values[0, mask1] = 1.0

    # Condition 2
    mask2 = (x1 >= 2.0) & (x0 <= 1.05) & (x1 < 2.03) & (x0 > 0.02)
    values[0, mask2] = 0.5

    # Condition 3
    mask3 = (x0 >= 1.05) & (x0 < 1.08) & (x1 < 2.03)
    values[0, mask3] = 0.5

    # Else: values remain 0.0
    return values


wh_n.sub(0).interpolate(y0_init)
wh_n.sub(2).interpolate(y1_init)
wh_n.x.scatter_forward()


v_function, p_function, c_function = wh.split()
vn, pn, cn = wh_n.split()

from turbidity_utils import (
    lock_Exchange_boundary_condition,)

bcs, facet_tags, ds = lock_Exchange_boundary_condition(domain, W, WIDTH, HEIGHT)

xdmf_solution.write_function(v_function, 0.0)
xdmf_solution.write_function(p_function, 0.0)
xdmf_solution.write_function(c_function, 0.0)

# Setup Variational Problem
dtt = fem.Constant(domain, DT)
h = ufl.CellDiameter(domain)
NU = fem.Constant(domain, 1 / GR)
F_FORCE = fem.Constant(domain, (0.0, -1.0))  # body forces
SED = fem.Constant(domain, (0.0, -1 * 0.00))  # sedimentation vel (SHOULD NOT BE CHANGED)
NU2 = fem.Constant(domain, 1 / (GR * SC))

vnorm = ufl.sqrt(ufl.dot(v_function, v_function))
# Use these in F instead of separate v_function, c_function
F = (
    ufl.inner(v_function - vn, w) / dtt
    + ufl.inner(ufl.dot(v_function, ufl.nabla_grad(v_function)), w)
    + NU * ufl.inner(ufl.grad(w), ufl.grad(v_function))
    - ufl.inner(p_function, ufl.div(w))
    + ufl.inner(q, ufl.div(v_function))
    - c_function * ufl.inner(F_FORCE, w)
) * ufl.dx + (
    ufl.inner(c_function - cn, qc) / dtt
    + ufl.inner(ufl.dot((v_function + SED), ufl.nabla_grad(c_function)), qc)
    + NU2 * ufl.inner(ufl.grad(qc), ufl.grad(c_function))
) * ufl.dx

# Then the stabilization terms, replacing v_function and c_function by these splits too
vnorm = ufl.sqrt(ufl.dot(v_function, v_function))
h = ufl.CellDiameter(domain)
R = (
    (1.0 / dtt) * (v_function - vn)
    + ufl.dot(v_function, ufl.nabla_grad(v_function))
    - NU * ufl.div(ufl.grad(v_function))
    + ufl.grad(p_function)
    - c_function * F_FORCE
)
R2 = (
    (1.0 / dtt) * (c_function - cn)
    + ufl.dot((v_function + SED), ufl.nabla_grad(c_function))
    - NU2 * ufl.div(ufl.grad(c_function))
)
tau = ((2.0 / dtt) ** 2 + (2.0 * vnorm / h) ** 2 + 9.0 * (4.0 * NU / (h * 2)) ** 2) ** (-0.5)

tau_lsic = (vnorm * h) / 2.0
F_lsic = tau_lsic * ufl.inner(ufl.div(v_function), ufl.div(w)) * ufl.dx
F_supg = tau * ufl.inner(R, ufl.dot(v_function, ufl.nabla_grad(w))) * ufl.dx
F_supg2 = tau * ufl.inner(R2, ufl.dot((v_function + SED), ufl.grad(qc))) * ufl.dx
F_pspg = tau * ufl.inner(R, ufl.grad(q)) * ufl.dx

F += F_supg + F_supg2 + F_pspg + F_lsic

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
opts[f"{option_prefix}pc_type"] = "hypre"
opts[f"{option_prefix}pc_hypre_type"] = "ilu"
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
    # After copying the arrays:
    wh_n.x.array[:] = wh.x.array[:]   
    wh_n.x.scatter_forward()        
    T = T + DT
    STEP += 1


    if STEP % EXPORT_INTERVAL == 0:
        v_function, p_function, c_function = wh.split()
        xdmf_solution.write_function(v_function, T)
        xdmf_solution.write_function(p_function, T)
        xdmf_solution.write_function(c_function, T)
        print(f"Exporting at t = {T:10.3e}...")


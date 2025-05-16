from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, io
import dolfinx.fem.petsc as petsc 
import numpy as np
import ufl
from pathlib import Path

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
# Personal note (Difference from FEniCS legacy): now we have to provide 
# the MPI-communicator as an argument to the built-in mesh creation

V = fem.functionspace(domain, ("Lagrange", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
# Personal note (Difference from FEniCS legacy): Expression is 
# interpolated directly as a lambda function

# Extract information from the mesh and function space
print(V.tabulate_dof_coordinates())
print(domain.geometry.x)

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
# Personal note (Difference from FEniCS legacy): We used to directly
# interpolate the Expression into the surface using on_boundary. Now
# we need to explicity extract the boundary facets.

# Now Dirichlet BCs can be defined
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# Define source term
f = fem.Constant(domain, default_scalar_type(-6.0))

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
# Personal note (from the Tutorial): dot should be used instead of inner
# for this problem given that we are using two rank one tensors. If we
# were using two rank two tensors, we would use inner.

# Assemble system
problem = petsc.LinearProblem(
    a, 
    L, 
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly", 
        "pc_type": "lu"
        }
    )
uh = fem.Function(V)
uh.name = "solution"
uh = problem.solve()

# Compute error from the manufactured solution
L2_error = fem.form(ufl.inner(uh - uD, uh - uD) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = np.max(np.abs(uD.x.array-uh.x.array))

if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Save solution to file in VTK format
with io.XDMFFile(domain.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(uh)


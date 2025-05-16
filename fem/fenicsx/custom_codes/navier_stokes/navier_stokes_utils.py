from dolfinx.fem import dirichletbc, Function, FunctionSpace, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import as_vector, FacetNormal
import numpy as np
import dolfinx.fem as fem
import ufl
from petsc4py import PETSc


    

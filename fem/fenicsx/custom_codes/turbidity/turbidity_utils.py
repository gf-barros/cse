from dolfinx.fem import dirichletbc, Function, FunctionSpace
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import as_vector, FacetNormal
import numpy as np
import dolfinx.fem as fem
import ufl

def lock_Exchange_boundary_condition(domain, W, width, height):
    """
    Boundary conditions for the 2D tank in FEniCSx.
    Applies no-slip on top/bottom, fixed u_x on left/right, and fixed scalar on the right.
    """
    # Define tags
    walls_marker = 1
    left_marker = 2
    right_marker = 3

    # Locate boundaries
    fdim = domain.topology.dim - 1  # facet dimension

    def walls(x):
        return np.isclose(x[1], 0.0) | np.isclose(x[1], height)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], width)

    def point_pressure(x):
        return np.isclose(x[0], width) & np.isclose(x[1], height)

    # Get boundary facets
    walls_facets = locate_entities_boundary(domain, fdim, walls)
    left_facets = locate_entities_boundary(domain, fdim, left)
    right_facets = locate_entities_boundary(domain, fdim, right)
    point_pressure_facets = locate_entities_boundary(domain, fdim, point_pressure)

    # Combine all for tagging
    all_facets = np.concatenate([walls_facets, left_facets, right_facets])
    all_markers = np.concatenate([
        np.full(len(walls_facets), walls_marker, dtype=np.int32),
        np.full(len(left_facets), left_marker, dtype=np.int32),
        np.full(len(right_facets), right_marker, dtype=np.int32)
    ])

    # Create facet tag
    sorted_facets = np.argsort(all_facets)
    facet_tag = meshtags(domain, fdim, all_facets[sorted_facets], all_markers[sorted_facets])

    # Create measures (optional, if you'll use ds or dS)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    # Constants
    u_noslip = fem.Constant(domain, (0.0, 0.0))      # For top and bottom
    u_zero_x = fem.Constant(domain, 0.0)             # For fixed u_x
    c_fixed_right = fem.Constant(domain, 0.0)        # For fixed scalar

    # FunctionSpaces for subcomponents
    Vx, _ = W.sub(0).sub(0).collapse()
    Vu, _ = W.sub(0).collapse()
    Q, _  = W.sub(2).collapse()

    # Interpolation functions
    bc_wall_func = fem.Function(Vu)
    bc_wall_func.interpolate(lambda x: np.zeros((2, x.shape[1])))

    bc_ux_func = fem.Function(Vx)
    bc_ux_func.interpolate(lambda x: np.zeros(x.shape[1]))

    bc_q_func = fem.Function(Q)
    bc_q_func.interpolate(lambda x: np.zeros(x.shape[1]))

    bc_p_func = fem.Function(Q)
    bc_p_func.interpolate(lambda x: np.zeros(x.shape[1]))

    # Apply DirichletBCs using extracted DOFs
    bc_1 = fem.dirichletbc(bc_ux_func, fem.locate_dofs_topological(Vx, fdim, left_facets))
    bc_2 = fem.dirichletbc(bc_ux_func, fem.locate_dofs_topological(Vx, fdim, right_facets))
    bc_3 = fem.dirichletbc(bc_wall_func, fem.locate_dofs_topological(Vu, fdim, walls_facets))
    bc_4 = fem.dirichletbc(bc_q_func, fem.locate_dofs_topological(Q, fdim, right_facets))
    bc_5 = fem.dirichletbc(bc_p_func, fem.locate_dofs_topological(Q, fdim, point_pressure_facets))

    return [bc_1, bc_2, bc_3, bc_4, bc_5], facet_tag, ds

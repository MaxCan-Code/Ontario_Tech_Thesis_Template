# from https://fenicsproject.org/olddocs/dolfin/latest/python/demos/singular-poisson
from dolfin import *

def singular_poisson(mesh, degree, κ, f, g, points):
    "Solve singular Poisson with point sources using supplied mesh and κ"

    # Then, we check that dolfin is configured with the backend called
    # PETSc, since it provides us with a wide range of methods used by
    # :py:class:`KrylovSolver <dolfin.cpp.la.KrylovSolver>`. We set PETSc as
    # our backend for linear algebra::

    # Test for PETSc
    if not has_linear_algebra_backend("PETSc"):
        info("DOLFIN has not been configured with PETSc. Exiting.")
        exit()

    parameters["linear_algebra_backend"] = "PETSc"

    # Create mesh and define function space
    V = FunctionSpace(mesh, "CG", degree)

    # specify trial functions (the unknowns) and test functions on the space :math:`V`
    u = TrialFunction(V)
    v = TestFunction(V)

    # With :math:`u,v,f` and :math:`g`, we can write down the bilinear form
    # :math:`a` and the linear form :math:`L` (using UFL operators). ::

    a = κ * inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # In order to transform our variational problem into a linear system we
    # need to assemble the coefficient matrix ``A`` and the right-side
    # vector ``b``. We do this using the function :py:meth:`assemble
    # <dolfin.cpp.fem.Assembler.assemble>`: ::

    # Assemble system
    A, b = assemble_system(a, L)

    # load and apply point source
    ps = PointSource(V, points)
    ps.apply(b)

    # We specify a Vector for storing the result by defining a
    # :py:class:`Function <dolfin.cpp.function.Function>`. ::

    # Solution Function
    u = Function(V)

    # Next, we specify the iterative solver we want to use, in this case a
    # :py:class:`PETScKrylovSolver <dolfin.cpp.la.PETScKrylovSolver>` with
    # the conjugate gradient (CG) method, and attach the matrix operator to
    # the solver. ::

    # Create Krylov solver
    solver = PETScKrylovSolver("cg")
    solver.set_operator(A)

    # We impose our additional constraint by removing the null space
    # component from the solution vector. In order to do this we need a
    # basis for the null space. This is done by creating a vector that spans
    # the null space, and then defining a basis from it. The basis is then
    # attached to the matrix ``A`` as its null space. ::

    # Create vector that spans the null space and normalize
    null_vec = Vector(u.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to PETSc matrix
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(A).set_nullspace(null_space)

    # Orthogonalization of ``b`` with respect to the null space makes sure
    # that it doesn't contain any component in the null space. ::

    null_space.orthogonalize(b)

    # Finally we are able to solve our linear system ::

    solver.solve(u.vector(), b)

    return u


def gen_refned_mesh(n_cells, n_refns, R, src_size, obj_padded):
    """
    Refine mesh close to plane and object
    R   radius of domain
    """
    side, fwd = src_size
    boarder = 2

    # Create mesh
    from mshr import Sphere, generate_mesh

    domain = Sphere(Point(), R)
    mesh = generate_mesh(domain, n_cells)
    # domain.set_subdomain(1, Sphere(p, a))

    # define subdomains
    charge_plane = CompiledSubDomain(
        f"abs(x[0]) <= { side + boarder } and "
        f"abs(x[1]) <= { fwd + boarder } and "
        f"abs(x[2]) <= { boarder * 2}"
    )

    for i in range(n_refns):
        # Mark cells for refinement
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)

        charge_plane.mark(cell_markers, True)
        obj_padded.mark(cell_markers, True)
        # CompiledSubDomain(charge_plane_mask).mark(cell_markers, True)

        # Refine mesh
        mesh = refine(mesh, cell_markers)

    return mesh


def mk_plane_source(size):
    "point sources based on evenly spaced mesh"

    # make mesh grid
    side, fwd = size
    corner = Point(-side / 2, -fwd / 2)
    mesh_coords = RectangleMesh(
        corner, Point() - corner, side - 1, fwd - 1
    ).coordinates()

    # q = 20 mV / cm
    n_src = side * fwd
    q = 20 / n_src
    q_tail = 20 / side
    q_all = (-q_tail,) * side + (q,) * (n_src - side)

    points = [(Point(loc), q_) for loc, q_ in zip(mesh_coords, q_all)]
    
    return points


def gen_obj(geom, a, d, boarder):
    if geom == "sphere":
        obj_mask = (
            f"pow(x[0] / 3, 2) + "
            f"pow(x[1], 2) + "
            f"pow(x[2] - {d}, 2) "
            f"<= {(a + boarder) ** 2}"
        )
    elif geom == "box":
        obj_mask = (
            f"abs(x[0]) <= {a + boarder} and "
            f"abs(x[1]) <= {a + boarder} and "
            f"abs(x[2] - {d}) <= {a + boarder}"
        )

    return CompiledSubDomain(obj_mask)


def coeff(mesh, obj):

    # Define subdomain markers
    markers = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

    # apply marker (bool mask)
    obj.mark(markers, 1)

    # Define magnetic permeability
    class Permeability(UserExpression):
        def __init__(self, markers, **kwargs):
            super().__init__(**kwargs)
            self.markers = markers

        def eval_cell(self, values, x, cell):
            # eps_e = 80.1, eps_i = 2.25,
            if self.markers[cell.index] == 1:
                values[0] = 2.25 / 80.1  # eps_i = 2.25
            else:
                values[0] = 1            # eps_e = 80.1

    κ = Permeability(markers, degree=1)

    return κ


def test_pert(obj_geom, n_cells, n_refns, R, points):
def test_singular_poisson():
    "test and plot"

    # source :math:`f` and boundary normal derivative :math:`g`	
    f = Constant(0)
    g = Constant(0)

    """
    object radius a = 2.0 cm
    d = 7 cm
    """
    a, d = 2, 7

    sphere_padded = gen_obj("box", a, d, boarder=2)
    sphere_ = gen_obj("box", a, d, boarder=0)

    plane_size = (4, 10)
    refned_mesh = gen_refned_mesh(30, 2, 50, plane_size, sphere_padded)

    κ_obj = coeff(refned_mesh, sphere_)

    u_obj = singular_poisson(refned_mesh, 1, κ_obj, f, g, points)

    u_empty = singular_poisson(refned_mesh, 1, Constant(1), f, g, points)
    
    return u_obj, u_empty


def plot_and_save(u_obj, u_empty):
    # and plot the solution ::
    "plot and save the solution"

    u_diff = u_obj - u_empty

    import matplotlib.pyplot as plt
    
    plt.colorbar(plot(u_empty)); plt.show()
    plt.colorbar(plot(u_diff)); plt.show()

    File("singular-poisson/solution_empty.pvd") << u_empty
    File("singular-poisson/solution_diff.pvd") << project(u_diff, u_obj.function_space())


def main():
    plane_size = (4, 10)
    p_mag = mk_plane_source(plane_size)
    # p_mag = [(Point(), 20)]

    u_obj, u_empty = test_pert("box", 30, 2, 50, p_mag)
    plot_and_save(u_obj, u_empty)


main()

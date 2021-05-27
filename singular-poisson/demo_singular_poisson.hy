(import dolfin *)

(defn singular-poisson [mesh degree κ f g points]
  "Solve singular Poisson with point sources using supplied mesh and κ
    from https://fenicsproject.org/olddocs/dolfin/latest/python/demos/singular-poisson
    and https://github.com/live-clones/dolfin/tree/master/python/demo/documented/singular-poisson"

  ;; Then, we check that dolfin is configured with the backend called
  ;; PETSc, since it provides us with a wide range of methods used by
  ;; :py:class:`KrylovSolver <dolfin.cpp.la.KrylovSolver>`. We set PETSc as
  ;; our backend for linear algebra::

  ;; Test for PETSc
  (if (not (has-linear-algebra-backend "PETSc"))
      (do (info "DOLFIN has not been configured with PETSc. Exiting.")
          (exit)))

  (setv (get parameters "linear_algebra_backend") "PETSc"

        ;; Create mesh and define function space
        V (FunctionSpace mesh "CG" degree)

        ;; specify trial functions (the unknowns) and test functions on the space :math:`V`
        u (TrialFunction V)
        v (TestFunction V)

        ;; With :math:`u,v,f` and :math:`g`, we can write down the bilinear form
        ;; :math:`a` and the linear form :math:`L` (using UFL operators). ::

        a (* κ (inner (grad u) (grad v)) dx)
        L (+ (* f v dx) (* g v ds))

        ;; In order to transform our variational problem into a linear system we
        ;; need to assemble the coefficient matrix ``A`` and the right-side
        ;; vector ``b``. We do this using the function :py:meth:`assemble
        ;; <dolfin.cpp.fem.Assembler.assemble>`: ::

        q 20
        u_e (Expression
              f"
        {q}/(4 * pi * sqrt(
        pow(x[0], 2)+
        pow(x[1], 2)+
        pow(x[2], 2)))
        "
              :degree 2)

        ;; Assemble system
        ;; u_D (Constant 0)
        ;; u_D u_e
        ;; bc (DirichletBC V u_D "on-boundary")
        ;; [A b] (assemble-system a L bc)
        [A b] (assemble-system a L)


        ;; load and apply point source
        ps (PointSource V points))
  (.apply ps b)

  ;; We specify a Vector for storing the result by defining a
  ;; :py:class:`Function <dolfin.cpp.function.Function>`. ::

  ;; Solution Function
  (setv u (Function V))

  ;; Next, we specify the iterative solver we want to use, in this case a
  ;; :py:class:`PETScKrylovSolver <dolfin.cpp.la.PETScKrylovSolver>` with
  ;; the conjugate gradient (CG) method, and attach the matrix operator to
  ;; the solver. ::

  ;; Create Krylov solver
  (setv solver (PETScKrylovSolver "cg"))
  (.set-operator solver A)

  ;; We impose our additional constraint by removing the null space
  ;; component from the solution vector. In order to do this we need a
  ;; basis for the null space. This is done by creating a vector that spans
  ;; the null space, and then defining a basis from it. The basis is then
  ;; attached to the matrix ``A`` as its null space. ::

  ;; Create vector that spans the null space and normalize
  (setv null-vec (Vector (.vector u)))
  (.set (.dofmap V) null-vec 1.0)
  (*= null-vec (/ 1.0 (.norm null-vec "l2")))
  ;; (/= null-vec (.norm null-vec "l2"))

  ;; Create null space basis object and attach to PETSc matrix
  (setv null-space (VectorSpaceBasis [null-vec]))
  (.set-nullspace (as-backend-type A) null-space)

  ;; Orthogonalization of ``b`` with respect to the null space makes sure
  ;; that it doesn't contain any component in the null space. ::

  (.orthogonalize null-space b)

  ;; Finally we are able to solve our linear system ::

  (.solve solver (.vector u) b)

  u)


(defn gen-refned-mesh [n-cells n-refns R src-size obj-padded]
  "
    Refine mesh close to plane and object
    R   radius of domain
    "
  (setv [side fwd] src-size
        boarder 2)

  ;; Create mesh
  (import mshr [Sphere generate-mesh])

  (setv domain (Sphere (Point) R)
        mesh (generate-mesh domain n-cells)
        ;; (.set-subdomain domain 1 (Sphere p a))

        ;; define subdomains
        charge-plane (CompiledSubDomain
                       f"abs(x[0]) <= {(+ side boarder)} and
          abs(x[1]) <= {(+ fwd boarder)} and
          abs(x[2]) <= {(* boarder 2)}"))

  (for [i (range n-refns)]
    ;; Mark cells for refinement
    (setv cell-markers (MeshFunction "bool" mesh (.dim (.topology mesh)) False))

    (.mark charge-plane cell-markers True)
    (.mark obj-padded cell-markers True)
    ;; (.mark (CompiledSubDomain charge-plane-mask) cell-markers True)

    ;; Refine mesh
    (setv mesh (refine mesh cell-markers)))

  mesh)


(defn mk-plane-source [size]
  "point sources based on evenly spaced mesh"

  ;; make mesh grid
  (setv [side fwd] size
        corner (Point (/ (- side) 2) (/ (- fwd) 2))
        mesh-coords (.coordinates (RectangleMesh
                                    corner (- (Point) corner) (- side 1) (- fwd 1)))

        ;; q = 20 mV / cm
        n-src (* side fwd)
        q (/ 20 n-src)
        q-tail (/ 20 side)
        q-all (+ (* [(- q-tail)] side) (* [q] (- n-src side)))

        points (lfor [loc q-] (zip mesh-coords q-all) [(Point loc) q-]))

  points)


(defn gen-obj [geom a d boarder]
                                ; todo : tol (+ a boarder)
  (setv obj-mask (match geom
                   "sphere"
                   f"(pow (/ x[0] 3) 2) +
              pow(x[1], 2) +
              pow(x[2] - {d}, 2)
              <= {(** (+ a boarder) 2)}"

                   "box"
                   f"abs(x[0]) <= {(+ a boarder)} and
              abs(x[1] - 2) <= {(+ a boarder)} and
              abs(x[2] - {d}) <= {(+ a boarder)}"))

  (CompiledSubDomain obj-mask))


(defn coeff [mesh obj]

  ;; Define subdomain markers
  (setv markers (MeshFunction "size_t" mesh (.dim (.topology mesh)) 0))

  ;; apply marker (bool mask)
  (.mark obj markers 1)

  ;; Define magnetic permeability
  (defclass Permeability [UserExpression]
    (defn __init__ [self markers &kwargs kwargs]
      (.__init__ (super) #** kwargs)
      (setv (. self markers) markers))

    (defn eval-cell [self values x cell]
      ;; eps_e = 80.1, eps_i = 2.25,
      (setv (get values 0)
            (if (= (get (. self markers) (. cell index)) 1)
                (/ 2.25 80.1) ;; eps_i = 2.25
                1))))         ;; eps_e = 80.1

  (setv κ (Permeability markers :degree 1))
  κ)


(defn test-pert [obj-geom n-cells n-refns R points]

  ;; source :math:`f` and boundary normal derivative :math:`g`
  (setv f (Constant 0)
        g (Constant 0)

        ;; object radius a = 2.0 cm
        ;; d = 7 cm

        [a d] [2 7]

        obj-padded (gen-obj obj-geom a d :boarder 2)
        obj- (gen-obj obj-geom a d :boarder 0)

        plane-size [4 10]
        refned-mesh (gen-refned-mesh n-cells n-refns R plane-size obj-padded)

        κ_obj (coeff refned-mesh obj-)

        u_obj (singular-poisson refned-mesh 1 κ_obj f g points)

        u_empty (singular-poisson refned-mesh 1 (Constant 1) f g points))

  (print "dofmap().global-dimension() = " (.dim (.function-space u_empty)))

  [u_obj u_empty])


(defn plot-and-save [u_obj u_empty]
  "plot and save the solution"

  (setv u_diff (- u_obj u_empty))

  (import matplotlib.pyplot :as plt)

  (.colorbar plt (plot u_empty)) (.show plt)
  (.colorbar plt (plot u_diff)) (.show plt)

  (<< (File "singular-poisson/solution_empty.pvd") u_empty)
  (<< (File "singular-poisson/solution_diff.pvd") (project u_diff (.function-space u_obj)))
  None)


(defn test-pert-pl-sv []
  (setv plane-size [4 10]
        ps-plane (mk-plane-source plane-size)

        [u_obj u_empty] (test-pert "box" 30 2 50 ps-plane))
  (plot-and-save u_obj u_empty)
  None)


;; %time (test-pert-pl-sv)

(import pandas :as pd
        numpy :as np)


(defn tup-df [dict- index-key cols-key]

  (setv cols-l (get dict- cols-key)
        index-l (get dict- index-key)

        tup-table (lfor j index-l (lfor i cols-l [i j]))

        index (.from-product (. pd MultiIndex) [[index-key] index-l] :names ["" ""])
        columns (.from-product (. pd MultiIndex) [[cols-key] cols-l] :names ["" ""]))

  (pd.DataFrame tup-table :index index :columns columns))


(defn val-map [x]
  (setv q 20
        u_e (Expression
              f"
        {q}/(4 * pi * sqrt(
        pow(x[0], 2)+
        pow(x[1], 2)+
        pow(x[2], 2)))
        "
              :degree 2)

        ps-pt [[(Point) q]]

        ;; [R- n-cells-] x
        ;; [u_obj u_empty] (test-pert "sphere" n-cells- 2 R- ps-pt)

        [n-refns- n-cells-] x
        [u_obj u_empty] (test-pert "sphere" n-cells- n-refns- 75 ps-pt)

        pt- [0 0 7]
        ;; show pert field at z=0 near body at 3 Rs
        pt-z0 [2 0 0])

  ;; (print "[R- n-cells-] =" x)
  (print "[n-refns- n-cells-] =" x)

  [(- (u_empty pt-z0) (u_obj pt-z0)) u_empty]

  [(- (u_empty pt-) (u_e pt-)) u_empty])


;; %time (print (val-map [1 40]))
;; %time (print (val-map [50 30]))

(defn sv-conv-table [params-d]
  (setv df (get (. (tup-df params-d "initial cells" "refinements") iloc) (, (slice None None -1)
                                                                            (slice None None -1)))
        ;; df (cut (. (tup-df params-d, "initial cells", "R") iloc) None None -1)

        df-mp (.applymap df val-map) ;; neu 10 min; dirl 9min 30

        df-fstr (cut (. (.applymap df-mp (fn [x] f"{(get x 0):.3e}")) iloc) None None -1)
        ;; (.applymap df-mp (fn [x] (.dim (.functionspace (get x 1)))))
        df-sv df-fstr)
  (.to-csv df-sv "out.csv")
  None)

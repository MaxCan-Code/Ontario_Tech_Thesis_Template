(import demo_singular_poisson :as dem)

(setv initial-cells-lst (lfor i (range 2 4) (* 8 (** 2 i))) ;; + [80]
      R-lst (lfor i (range 2 5) (* i 25))
      refn-lst (range 1 3)

      params-d {"initial cells" initial-cells-lst
                  "R" R-lst
                  "refinements" refn-lst})

(.sv-conv-table dem params-d)

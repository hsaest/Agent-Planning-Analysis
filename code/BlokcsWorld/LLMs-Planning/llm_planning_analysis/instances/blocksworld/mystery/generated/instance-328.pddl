(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g h i j k d)
(:init 
(harmony)
(planet g)
(planet h)
(planet i)
(planet j)
(planet k)
(planet d)
(province g)
(province h)
(province i)
(province j)
(province k)
(province d)
)
(:goal
(and
(craves g h)
(craves h i)
(craves i j)
(craves j k)
(craves k d)
)))
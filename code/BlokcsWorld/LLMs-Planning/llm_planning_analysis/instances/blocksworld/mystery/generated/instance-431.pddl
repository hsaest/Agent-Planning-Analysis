(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j d g k e)
(:init 
(harmony)
(planet j)
(planet d)
(planet g)
(planet k)
(planet e)
(province j)
(province d)
(province g)
(province k)
(province e)
)
(:goal
(and
(craves j d)
(craves d g)
(craves g k)
(craves k e)
)))
(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b d g c e f i a h j l)
(:init 
(harmony)
(planet b)
(planet d)
(planet g)
(planet c)
(planet e)
(planet f)
(planet i)
(planet a)
(planet h)
(planet j)
(planet l)
(province b)
(province d)
(province g)
(province c)
(province e)
(province f)
(province i)
(province a)
(province h)
(province j)
(province l)
)
(:goal
(and
(craves b d)
(craves d g)
(craves g c)
(craves c e)
(craves e f)
(craves f i)
(craves i a)
(craves a h)
(craves h j)
(craves j l)
)))
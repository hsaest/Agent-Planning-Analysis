(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c j i d h k b a f e l g)
(:init 
(harmony)
(planet c)
(planet j)
(planet i)
(planet d)
(planet h)
(planet k)
(planet b)
(planet a)
(planet f)
(planet e)
(planet l)
(planet g)
(province c)
(province j)
(province i)
(province d)
(province h)
(province k)
(province b)
(province a)
(province f)
(province e)
(province l)
(province g)
)
(:goal
(and
(craves c j)
(craves j i)
(craves i d)
(craves d h)
(craves h k)
(craves k b)
(craves b a)
(craves a f)
(craves f e)
(craves e l)
(craves l g)
)))
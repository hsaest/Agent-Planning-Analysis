(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a j e i c b f)
(:init 
(harmony)
(planet a)
(planet j)
(planet e)
(planet i)
(planet c)
(planet b)
(planet f)
(province a)
(province j)
(province e)
(province i)
(province c)
(province b)
(province f)
)
(:goal
(and
(craves a j)
(craves j e)
(craves e i)
(craves i c)
(craves c b)
(craves b f)
)))
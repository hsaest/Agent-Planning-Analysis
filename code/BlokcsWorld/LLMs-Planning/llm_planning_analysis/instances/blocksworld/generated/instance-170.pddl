(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g c h i e j a f b)
(:init 
(handempty)
(ontable g)
(ontable c)
(ontable h)
(ontable i)
(ontable e)
(ontable j)
(ontable a)
(ontable f)
(ontable b)
(clear g)
(clear c)
(clear h)
(clear i)
(clear e)
(clear j)
(clear a)
(clear f)
(clear b)
)
(:goal
(and
(on g c)
(on c h)
(on h i)
(on i e)
(on e j)
(on j a)
(on a f)
(on f b)
)))
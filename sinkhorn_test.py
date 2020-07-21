from jax import numpy as np
import jax.random as random
from loss_funcs import sinkhorn_loss
from datetime import datetime
import time

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)


start_time_initialization = time.time()

M = 40

P = random.normal(random.PRNGKey(0), (M, M))
Pt = random.normal(random.PRNGKey(1), (M, M))

sinkhorn_dist = sinkhorn_loss(P, Pt, M, eps=1e-3, max_iters=100, stop_thresh=1e-3)

print("Sinkhorn Distanc: %f" % (sinkhorn_dist))
print("running time: %s seconds ---" % (time.time() - start_time_initialization))
exit()

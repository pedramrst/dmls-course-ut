from mpi4py import MPI
import random
import time
from decimal import Decimal, getcontext

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
i = 40000000
counter = 0
getcontext().prec = 40
if rank == 0:
    start_time = time.time()
for _ in range(int(i // size)):
    a, b = random.uniform(-1, 1), random.uniform(-1, 1)
    if (a ** 2 ) + (b ** 2) <= 1:
        counter += 1

all_counters = comm.reduce(counter, root=0)
if rank == 0:
    print(f"PI: {Decimal(4 * all_counters / i)}")
    print(f"time: {time.time() - start_time} (s)")

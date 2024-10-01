import random
import time
from decimal import Decimal, getcontext


i = 40_000_000
counter = 0
getcontext().prec = 40
start_time = time.time()
for _ in range(i):
    a, b = random.uniform(-1, 1), random.uniform(-1, 1)
    if (a ** 2 ) + (b ** 2) <= 1:
        counter += 1
print(f"PI: {Decimal(4 * counter / i)}")
print(f"time: {time.time() - start_time} (s)")

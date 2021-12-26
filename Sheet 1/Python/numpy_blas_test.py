import numpy as np
import time

np.__config__.show()

n = 10000
k = 100

A = np.random.random( (n,n) )
B = np.random.random( (n,k) )
C = np.zeros( (n,k) )

t = time.time()
for j in range(k):
    for i in range(n):
        C[i,j] = A[[i],:]@B[:,[j]]

level1_time = time.time() - t
print("BLAS LEVEL 1:", level1_time, "Sekunden")

t = time.time()
for j in range(k):
    C[:,j] = A@B[:,j]

level2_time = time.time() - t
print("BLAS LEVEL 2:", level2_time, "Sekunden")

t = time.time()
C = A@B
level3_time = time.time() - t
print("BLAS LEVEL 3:", level3_time, "Sekunden")

print("speedup 1 vs. 2:", level1_time/level2_time)
print("speedup 2 vs. 3:", level2_time/level3_time)
print("speedup 1 vs. 3:", level1_time/level3_time)
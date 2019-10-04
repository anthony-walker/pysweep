from concurrent import futures
from time import time
import multiprocessing as mp
import numpy as np
import mpi4py as MPI



def test_fcn(x):
    print("Here")
    for i in range(1000):
        x += 0.0001
    return x




# start = time()
# tpool = futures.ThreadPoolExecutor()
# tl = np.arange(0,10000,1)
# future1 = [tpool.submit(test_fcn,item) for item in tl]
# futures.wait(future1)
# stop = time()
# print(stop-start)
# print(tl)

# start = time()
# ppool = mp.Pool()
# tl = np.arange(0,10000,1)
# future1 = ppool.map_async(test_fcn,tl)
# future1.get()
# stop = time()
# print(stop-start)
node_comm = MPI.Intracomm.Spawn()


from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
processor = MPI.Get_processor_name()
master_rank = ZERO #master rank
num_ranks = comm.Get_size() #number of ranks
rank = comm.Get_rank()  #current rank
print(rank,processor)
comm.Barrier()
#Test shared array creation
dT = np.float32
arr = np.zeros((10,10),dtype=dT)
ashape = arr.shape
win = MPI.Win.Allocate_shared(arr.nbytes, dT.itemsize, comm=comm)
shared_buf, itemsize = win.Shared_query(0)
sarr = np.ndarray(buffer=shared_buf, dtype=dT.type, shape=ashape)
comm.Barrier()
if rank == 0:
    sarr[:,:] = 3
comm.Barrier()
if rank == 1:
    print(sarr)
comm.Barrier()

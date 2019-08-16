from mpi4py import MPI
import sys
import numpy as np
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()  #current rank
#
# #Creating New Group
# rank_group = comm.group.Incl((0,1,2))
# comm = comm.Create_group(rank_group)
#
# if rank >= 3:
#     MPI.Finalize()
#     exit(0)
# comm.Barrier()
# print(rank)
# TOPS = 4
# ONE = 1
# block_size = (18,18,1)
# #Max Swept Steps from block_size and other constants
# min_block = min(block_size[:-1])
# MPSS = 1 #Max Pyramid Swept Step #Start at 1 to account for ghost node calc
# while(min_block>=(TOPS+ONE)):
#     min_block -= TOPS
#     MPSS += 1
TA = np.zeros((10,10))
TA[8:10,:] = 2
print(TA[8:10,:])
print(TA[-2:,:])

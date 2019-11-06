#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
from .dcore import sgs, decomp
from ccore import printer
#CUDA Imports
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
import numpy as np


def Decomposition(sarr,garr,blocks,gts,pargs,mpi_pool,total_cpu_block,gwrite):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:

        arr_gpu = cuda.mem_alloc(garr.nbytes)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("Decomp")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        sarr[gwrite]=garr[:,:,OPS:-OPS,OPS:-OPS]
    else:   #CPUs do this
        mpi_pool.map(dCPU_Decomp,blocks)
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = sgs.carr[:,:,OPS:-OPS,OPS:-OPS]

def dCPU_Decomp(block):
    """Use this function to build the Up Pyramid."""
    #Calculating Step
    ct = sgs.gts
    sgs.carr[block] = sgs.SM.step(sgs.carr[block],sgs.dset,1,ct)

def send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,total_cpu_block,ops,gread,garr):
    """Use this function to communicate data between nodes"""
    if NMB:
        sarr[:,:,ops:-ops,:ops] = sarr[:,:,ops:-ops,-2*ops:-ops]
        sarr[:,:,ops:-ops,-ops:] = sarr[:,:,ops:-ops,ops:2*ops]
        bufffor = np.copy(sarr[:,:,-2*ops:-ops,:])
        buffback = np.copy(sarr[:,:,ops:2*ops,:])
        bufferf = cluster_comm.sendrecv(sendobj=bufffor,dest=comranks[1],source=comranks[0])
        bufferb = cluster_comm.sendrecv(sendobj=buffback,dest=comranks[0],source=comranks[1])
        cluster_comm.Barrier()
        sarr[:,:,:ops,:] = bufferf[:,:,:,:]
        sarr[:,:,-ops:,:] = bufferb[:,:,:,:]
    node_comm.Barrier()

    if not GRB:
        sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    else:
        garr[:,:,:,:] = sarr[gread]

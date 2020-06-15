#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
from ddcore import sgs, decomp,cpu
from ccore import printer
#CUDA Imports
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
import numpy as np


def Decomposition(GRB,OPS,sarr,garr,blocks,mpi_pool,DecompObj):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Splitting between cpu and gpu
    if GRB:
        DecompObj(garr)
        sarr[DecompObj.gwrite]=garr[:,:,OPS:-OPS,OPS:-OPS]
    else:   #CPUs do this
        mpi_pool.map(DecompObj,blocks)
        #Copy result to MPI shared process array
        sarr[DecompObj.cwrite] = sgs.carr[:,:,OPS:-OPS,OPS:-OPS]
    DecompObj.gts+=1 #Update global time step

def send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,ops,garr,DecompObj):
    """Use this function to communicate data between nodes"""
    if NMB:
        sarr[:,:,ops:-ops,:ops] = sarr[:,:,ops:-ops,-2*ops-1:-ops-1]
        sarr[:,:,ops:-ops,-ops:] = sarr[:,:,ops:-ops,1:ops+1]
        bufffor = np.copy(sarr[:,:,-2*ops-1:-ops-1,:])
        buffback = np.copy(sarr[:,:,ops+1:2*ops+1,:])
        bufferf = cluster_comm.sendrecv(sendobj=bufffor,dest=comranks[1],source=comranks[0])
        bufferb = cluster_comm.sendrecv(sendobj=buffback,dest=comranks[0],source=comranks[1])
        cluster_comm.Barrier()
        sarr[:,:,:ops,:] = bufferf[:,:,:,:]
        sarr[:,:,-ops:,:] = bufferb[:,:,:,:]
    node_comm.Barrier()

    if GRB:
        garr[:,:,:,:] = sarr[DecompObj.gread]
    else:
        sgs.carr[:,:,:,:] = sarr[DecompObj.cread]


class DecompCPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args,cwrite=None,cread=None):
        """Use this function to create the initial function object"""
        self.gts,self.set,self.start = args
        self.cwrite=cwrite
        self.cread=cread

    def __call__(self,block):
        """Use this function to build the Up Pyramid."""
        sgs.carr[block] = cpu.step(sgs.carr[block],self.set,self.start,self.gts)

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompCPU,(self.gts,self.set,self.start))

class DecompGPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args):
        """Use this function to create the initial function object"""
        self.gts,self.gfunction,self.GRD,self.BS,self.gread,self.gwrite,garr = args
        self.arr_gpu = cuda.mem_alloc(garr.nbytes)


    def __call__(self,garr):
        """Use this function to build the Up Pyramid."""
        cuda.memcpy_htod(self.arr_gpu,garr)
        self.gfunction(self.arr_gpu,np.int32(self.gts),grid=self.GRD, block=self.BS)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,self.arr_gpu)
        return garr

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompGPU,(self.gts,self.set))

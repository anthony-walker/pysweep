#Programmer: Anthony Walker
#This file contains classes that act as functions in order to maintain the state of the functions
#Package imports
from dcore import block, decomp, cpu, sgs
from ccore import source
#Other imports
import numpy as np
#CUDA Imports
try:
    import pycuda.driver as cuda
except Exception as e:
    pass
    # print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")



def FirstPrism(*args):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    # SM,GRB,BS,GRD,OPS,TSO,ssb = pargs
    SM,GRB,Up,Yb,mpiPool,blocks,sarr,garr,total_cpu_block = args
    #Splitting between cpu and gpu
    if GRB:
        lgarr = decomp.get_local_extend_array(sarr,np.zeros(Up.shape),blocks,Up.BS)
        cuda.memcpy_htod(garr,lgarr)
        Up(garr)
        Yb(garr)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(lgarr,garr)
        sarr[blocks]=lgarr[:,:,:,Up.BS[0]:-Up.BS[0]]
        Up.gts+=Up.MPSS
        Yb.gts+=Yb.MPSS
        Yb.gfunction = SM.get_function("YBT") #Change to new ybridge function
    else:   #CPUs do this
        mpiPool.map(Up,blocks[0])
        mpiPool.map(Yb,blocks[1])
        Up.gts+=Up.MPSS
        Yb.gts+=Yb.MPSS
        Yb.start += Yb.MPSS #Change starting location for UpPrism
        sarr[total_cpu_block] = sgs.carr[:,:,:,:]

def UpPrism(*args):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    GRB,Xb,Yb,Oct,mpiPool,blocks,sarr,garr,total_cpu_block = args
    #Splitting between cpu and gpu
    if GRB:

        lgarr = decomp.get_local_extend_array(sarr,np.zeros(Oct.shape),blocks,Oct.BS)
        cuda.memcpy_htod(garr,lgarr)
        Xb(garr)
        Oct(garr)
        Yb(garr)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(lgarr,garr)
        sarr[blocks]=lgarr[:,:,:,Oct.BS[0]:-Oct.BS[0]]
    else:   #CPUs do this
        mpiPool.map(Xb,blocks[0])
        mpiPool.map(Oct,blocks[1])
        mpiPool.map(Yb,blocks[0])
        sarr[total_cpu_block] = sgs.carr[:,:,:,:]
    Xb.gts+=Xb.MPSS
    Oct.gts+=Oct.MPSS
    Yb.gts+=Yb.MPSS

def LastPrism(*args):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    GRB,Xb,Down,mpiPool,blocks,sarr,garr,total_cpu_block = args
    #Splitting between cpu and gpu
    if GRB:
        lgarr = decomp.get_local_extend_array(sarr,np.zeros(Down.shape),blocks,Down.BS)
        cuda.memcpy_htod(garr,lgarr)
        Xb(garr)
        Down(garr)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(lgarr,garr)
        sarr[blocks]=lgarr[:,:,:,Down.BS[0]:-Down.BS[0]]
    else:   #CPUs do this
        mpiPool.map(Xb,blocks[0])
        mpiPool.map(Down,blocks[1])
        sarr[total_cpu_block] = sgs.carr[:,:,:,:]

def first_forward(NMB,GRB,node_comm,cluster_comm,comranks,sarr,spx,total_cpu_block):
    """Use this function to communicate data between nodes"""
    if NMB:
        buff = np.copy(sarr[:,:,-spx:,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[1],source=comranks[0])
        cluster_comm.Barrier()
        sarr[:,:,spx:,:] = sarr[:,:,:-spx,:] #Shift sarr data forward by spx
        sarr[:,:,:spx,:] = buffer[:,:,:,:]
    node_comm.Barrier()
    if not GRB:
        sgs.carr[:,:,:,:] = sarr[total_cpu_block]

def send_forward(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,spx,gts,TSO,MPSS,total_cpu_block):
    """Use this function to communicate data between nodes"""
    if NMB:
        cwt = decomp.swept_write(cwt,sarr,hdf5_data,gsc,gts,TSO,MPSS)
        buff = np.copy(sarr[:,:,-spx:,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[1],source=comranks[0])
        cluster_comm.Barrier()
        sarr[:,:,spx:,:] = sarr[:,:,:-spx,:] #Shift sarr data forward by spx
        sarr[:,:,:spx,:] = buffer[:,:,:,:]
    node_comm.Barrier()
    if not GRB:
        sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    return cwt

def send_backward(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,spx,gts,TSO,MPSS,total_cpu_block):
    """Use this function to communicate data between nodes"""
    if NMB:
        buff = np.copy(sarr[:,:,:spx,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[0],source=comranks[1])
        cluster_comm.Barrier()
        sarr[:,:,:-spx,:] = sarr[:,:,spx:,:] #Shift sarr backward data by spx
        sarr[:,:,-spx:,:] = buffer[:,:,:,:]
        cwt = decomp.swept_write(cwt,sarr,hdf5_data,gsc,gts,TSO,MPSS)
    node_comm.Barrier()
    if not GRB:
        sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    return cwt

class GeometryCPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args):
        """Use this function to create the initial function object"""
        self.gts,self.sets,self.tso,self.MPSS,self.start = args

    def __call__(self,block):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        ct = self.gts
        for ts,swept_set in enumerate(self.sets,start=self.start):
            #Calculating Step
            sgs.carr[block] = cpu.step(sgs.carr[block],swept_set,ts,ct)
            ct+=1
        return block

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (GeometryCPU,(self.gts,self.sets,self.tso,self.MPSS,self.start))


class GeometryGPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args):
        """Use this function to create the initial function object"""
        self.gts,self.gfunction,self.BS,self.GRD,self.MPSS,self.shape = args
    def __call__(self,garr):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(garr,np.int32(self.gts),grid=self.GRD, block=self.BS)

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (GeometryGPU,(self.gts,self.gfunction,self.BS,self.GRD,self.MPSS,self.shape))

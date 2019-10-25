#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
from .dcore import sgs, decomp
#CUDA Imports
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
import numpy as np

def FirstPrism(sarr,garr,blocks,gts,pargs,mpi_pool,total_cpu_block):
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
        garr = decomp.copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        res = mpi_pool.map(dCPU_UpPyramid,cblocks)
        mpi_pool.map(dCPU_Ybridge,xblocks)
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = sgs.carr[:,:,:,:]

def UpPrism(sarr,garr,blocks,up_sets,x_sets,gts,pargs,mpi_pool,total_cpu_block):
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
        garr = decomp.copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        printer.pm(garr,2,'%.0f')
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        mpi_pool.map(dCPU_UpPyramid,cblocks)
        mpi_pool.map(dCPU_Ybridge,xblocks)
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = sgs.carr[:,:,:,:]


def dCPU_UpPyramid(block):
    """Use this function to build the Up Pyramid."""
    #UpPyramid of Swept Step
    ct = sgs.gts
    print(block)
    for ts,swept_set in enumerate(sgs.up_sets,start=sgs.TSO-1):
        #Calculating Step
        sgs.carr[block] = sgs.SM.step(sgs.carr[block],swept_set,ts,ct)
        ct+=1

def dCPU_Ybridge(block):
    """Use this function to build the XBridge."""
    ct = sgs.gts
    for ts,swept_set in enumerate(sgs.y_sets,start=sgs.TSO-1):
        #Calculating Step
        sgs.carr[block] = sgs.SM.step(sgs.carr[block],swept_set,ts,ct)
        ct+=1

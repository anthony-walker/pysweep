#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
import sys
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from itertools import repeat

def UpPyramid(sarr,arr,WR,BDR,isets,gts,pargs):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,CRS,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        # Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("UpPyramid")
        gpu_fcn(cuda.InOut(arr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in CRS:
            block = np.copy(arr[local_region])
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    np.copyto(sarr[WR], arr[:,:,OPS:-OPS,OPS:-OPS])
    for br in BDR:
        np.copyto(sarr[br[0],br[1],br[4],br[5]],arr[br[0],br[1],br[2],br[3]])

def Bridge(sarr,xarr,yarr,XR,YR,isets,gts,pargs):
    """Use this function to solve the bridge step."""
    SM,GRB,BS,GRD,CRS,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        # X-Bridge
        xarr = np.ascontiguousarray(xarr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("BridgeX")
        gpu_fcn(cuda.InOut(xarr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
        # Y-Bridge
        yarr = np.ascontiguousarray(yarr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("BridgeY")
        gpu_fcn(cuda.InOut(yarr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks_x = []
        blocks_y = []
        for local_region in CRS:
            #X blocks
            block_x = np.copy(xarr[local_region])
            blocks_x.append(block_x)
            #Y_blocks
            block_y = np.copy(yarr[local_region])
            blocks_y.append(block_y)
        #Solving
        cpu_fcn = sweep_lambda((CPU_Bridge,SM,isets[0],gts,TSO))
        blocks_x = list(map(cpu_fcn,blocks_x))
        cpu_fcn.args = (CPU_Bridge,SM,isets[1],gts,TSO)
        blocks_y = list(map(cpu_fcn,blocks_y))
        xarr = rebuild_blocks(xarr,blocks_x,CRS,OPS)
        yarr = rebuild_blocks(yarr,blocks_y,CRS,OPS)
    for i, bdg in enumerate(XR,start=TSO+1):
        for cb  in bdg:
            for lcx,lcy,shx,shy in cb:
                np.copyto(sarr[i,:,shx,shy],xarr[i,:,lcx,lcy])
    for i,bdg in enumerate(YR,start=TSO+1):
        for cb  in bdg:
            for lcx,lcy,shx,shy in cb:
                np.copyto(sarr[i,:,shx,shy],yarr[i,:,lcx,lcy])

def Octahedron(sarr,arr,WR,BDR,isets,gts,pargs):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,CRS,OPS,TSO,ssb = pargs
    if GRB:
        #Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("Octahedron")
        gpu_fcn(cuda.InOut(arr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in CRS:
            block = np.copy(arr[local_region])
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_Octahedron,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    np.copyto(sarr[WR], arr[:,:,OPS:-OPS,OPS:-OPS])
    for br in BDR:
        np.copyto(sarr[br[0],br[1],br[4],br[5]],arr[br[0],br[1],br[2],br[3]])

def DownPyramid(sarr,arr,WR,BDR,isets,gts,pargs):
    """This is the ending inverted pyramid."""
    SM,GRB,BS,GRD,CRS,OPS,TSO,ssb = pargs
    if GRB:
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("DownPyramid")
        gpu_fcn(cuda.InOut(arr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in CRS:
            block = np.copy(arr[local_region])
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_DownPyramid,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    np.copyto(sarr[WR], arr[:,:,OPS:-OPS,OPS:-OPS])
    for br in BDR:
        np.copyto(sarr[br[0],br[1],br[4],br[5]],arr[br[0],br[1],br[2],br[3]])

#--------------------------------CPU Specific Swept Functions------------------
def CPU_UpPyramid(args):
    """Use this function to build the Up Pyramid."""
    block,SM,isets,gts,TSO = args
    #Removing elements for swept step
    for ts,swept_set in enumerate(isets,start=TSO-1):
        #Calculating Step
        block = SM.step(block,swept_set,ts,gts)
        gts+=1
    return block

def CPU_Bridge(args):
    """Use this function to build the Up Pyramid."""
    block,SM,isets,gts,TSO = args
    #Removing elements for swept step
    for ts,swept_set in enumerate(isets,start=TSO):
        #Calculating Step
        block = SM.step(block,swept_set,ts,gts)
        gts+=1
    return block

def CPU_Octahedron(args):
    """Use this function to build the Octahedron."""
    block,SM,isets,gts,TSO = args
    #Oct Step
    for ts,swept_set in enumerate(isets,start=TSO):
        #Calculating Step
        block = SM.step(block,swept_set,ts,gts)
        gts+=1
    return block

def CPU_DownPyramid(args):
    """Use this function to build the Down Pyramid."""
    block,SM,isets,gts,TSO = args
    #Removing elements for swept step
    for ts, swept_set in enumerate(isets,start=TSO):
        #Calculating Step
        block = SM.step(block,swept_set,ts,gts)
        gts+=1
    return block

#-----------------------------------dsweep functions below here------------------------------#

def FirstPrism(sarr,garr,blocks,total_cpu_block,up_sets,x_sets,gts,mpi_pool,pargs):
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
        garr = copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        mpi_pool.map(dCPU_UpPyramid,zip(cblocks,repeat(gts)))
        mpi_pool.map(dCPU_Ybridge,zip(xblocks,repeat(gts)))
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = carr[:,:,:,:]

def UpPrism(sarr,garr,blocks,total_cpu_block,up_sets,x_sets,gts,mpi_pool,pargs):
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
        garr = copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        pm(garr,2,'%.0f')
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        mpi_pool.map(dCPU_UpPyramid,zip(cblocks,repeat(gts)))
        mpi_pool.map(dCPU_Ybridge,zip(xblocks,repeat(gts)))
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = carr[:,:,:,:]


def dCPU_UpPyramid(block):
    """Use this function to build the Up Pyramid."""
    #UpPyramid of Swept Step
    global carr
    block,ct = block
    for ts,swept_set in enumerate(up_sets,start=TSO-1):
        #Calculating Step
        carr[block] = SM.step(carr[block],swept_set,ts,ct)
        ct+=1

def dCPU_Ybridge(block):
    """Use this function to build the XBridge."""
    global carr,TSO
    block,ct = block
    for ts,swept_set in enumerate(x_sets,start=TSO-1):
        #Calculating Step
        carr[block] = SM.step(carr[block],swept_set,ts,ct)
        ct+=1

#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
import sys
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .pysweep_block import *

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
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)

    sarr[WR] = arr[:,:,OPS:-OPS,OPS:-OPS]
    for br in BDR:
        sarr[br[0],br[1],br[4],br[5]] = arr[br[0],br[1],br[2],br[3]]

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
        # print(xarr[4,0,:,:])
    else:   #CPUs do this
        blocks_x = []
        blocks_y = []
        for local_region in CRS:
            #X blocks
            block_x = np.zeros(xarr[local_region].shape)
            block_x += xarr[local_region]
            blocks_x.append(block_x)
            #Y_blocks
            block_y = np.zeros(yarr[local_region].shape)
            block_y += yarr[local_region]
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
                sarr[i,:,shx,shy] = xarr[i,:,lcx,lcy]
    for i,bdg in enumerate(YR,start=TSO+1):
        for cb  in bdg:
            for lcx,lcy,shx,shy in cb:
                sarr[i,:,shx,shy] = yarr[i,:,lcx,lcy]

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
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_Octahedron,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    sarr[WR] = arr[:,:,OPS:-OPS,OPS:-OPS]
    for br in BDR:
        sarr[br[0],br[1],br[4],br[5]] = arr[br[0],br[1],br[2],br[3]]

def DownPyramid(sarr,arr,WR,isets,gts,pargs):
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
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_DownPyramid,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    sarr[WR] = arr[:,:,OPS:-OPS,OPS:-OPS]

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

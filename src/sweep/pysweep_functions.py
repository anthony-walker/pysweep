#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
import sys
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .pysweep_block import *

def UpPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,bregions,cpu_regions,shared_arr,idx_sets,ops,gts):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    #Finding splits again
    SPLITX = block_size[0]/2
    SPLITY = block_size[1]/2
    #Splitting between cpu and gpu

    if gpu_rank:
        # Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("UpPyramid")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,source_mod,idx_sets,gts))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,cpu_regions,ops)
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]
    for br in bregions:
        shared_arr[br[0],br[1],br[4],br[5]] = arr[br[0],br[1],br[2],br[3]]

def Bridge(source_mod,xarr,yarr,gpu_rank,block_size,grid_size,xregion,yregion,cregions,cpu_regions,shared_arr,idx_sets,ops,gts):
    """Use this function to solve the bridge step."""
    #Finding splits again
    SPLITX = block_size[0]/2
    SPLITY = block_size[1]/2
    y0arr = np.zeros(yarr.shape)
    y0arr[:,:,:,:] = yarr[:,:,:,:]
    x0arr = np.zeros(xarr.shape)
    x0arr[:,:,:,:] = xarr[:,:,:,:]
    #Splitting between cpu and gpu
    if gpu_rank:
        # X-Bridge
        xarr = np.ascontiguousarray(xarr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("BridgeX")
        ss = np.zeros(xarr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(xarr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
        # Y-Bridge
        yarr = np.ascontiguousarray(yarr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("BridgeY")
        ss = np.zeros(yarr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(yarr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks_x = []
        blocks_y = []

        for local_region in cpu_regions:
            #X blocks
            block_x = np.zeros(xarr[local_region].shape)
            block_x += xarr[local_region]
            blocks_x.append(block_x)
            #Y_blocks
            block_y = np.zeros(yarr[local_region].shape)
            block_y += yarr[local_region]
            blocks_y.append(block_y)
        #Solving
        cpu_fcn = sweep_lambda((CPU_Bridge,source_mod,idx_sets[0],gts))
        blocks_x = list(map(cpu_fcn,blocks_x))
        cpu_fcn.args = (CPU_Bridge,source_mod,idx_sets[1],gts)
        blocks_y = list(map(cpu_fcn,blocks_y))
        xarr = rebuild_blocks(xarr,blocks_x,cpu_regions,ops)
        yarr = rebuild_blocks(yarr,blocks_y,cpu_regions,ops)
    yarr-=y0arr
    xarr-=x0arr
    shared_arr[xregion] += xarr[:,:,:,:]
    shared_arr[yregion] += yarr[:,:,:,:]

    for i, x_item in enumerate(cregions[1],start=1):
        for bs in x_item:
            lcx,lcy,shx,shy = bs
            shared_arr[i,:,shx,shy] = xarr[i,:,lcx,lcy]
    for i, x_item in enumerate(cregions[0],start=1):
        for bs in x_item:
            lcx,lcy,shx,shy = bs
            shared_arr[i,:,shx,shy] = yarr[i,:,lcx,lcy]

def Octahedron(source_mod,arr,gpu_rank,block_size,grid_size,region,bregions,cpu_regions,shared_arr,idx_sets,ops,gts):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        #Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("Octahedron")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_Octahedron,source_mod,idx_sets,gts))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,cpu_regions,ops)
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]
    for br in bregions:
        shared_arr[br[0],br[1],br[4],br[5]] = arr[br[0],br[1],br[2],br[3]]

def DownPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,cpu_regions,shared_arr,idx_sets,ops,gts):
    """This is the ending inverted pyramid."""
    if gpu_rank:
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("DownPyramid")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_DownPyramid,source_mod,idx_sets,gts))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,cpu_regions,ops)
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

#--------------------------------CPU Specific Swept Functions------------------
def CPU_UpPyramid(args):
    """Use this function to build the Up Pyramid."""
    block,source_mod,idx_sets,gts = args
    #Removing elements for swept step
    for ts,swept_set in enumerate(idx_sets,start=0):
        #Calculating Step
        block = source_mod.step(block,swept_set,ts,gts)
    return block

def CPU_Bridge(args):
    """Use this function to build the Up Pyramid."""
    block,source_mod,idx_sets,gts = args
    #Removing elements for swept step
    for ts,swept_set in enumerate(idx_sets,start=0):
        #Calculating Step
        block = source_mod.step(block,swept_set,ts,gts)
    return block

def CPU_Octahedron(args):
    """Use this function to build the Octahedron."""
    block,source_mod,idx_sets,gts = args
    #Oct Step
    for ts,swept_set in enumerate(idx_sets,start=0):
        #Calculating Step
        block = source_mod.step(block,swept_set,ts,gts)
    return block

def CPU_DownPyramid(args):
    """Use this function to build the Down Pyramid."""
    block,source_mod,idx_sets,gts = args

    #Removing elements for swept step
    for ts, swept_set in enumerate(idx_sets,start=0):
        #Calculating Step
        block = source_mod.step(block,swept_set,ts,gts)
    return block

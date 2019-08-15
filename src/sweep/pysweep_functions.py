#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .pysweep_decomposition import *
def UpPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,cpu_regions,shared_arr,idx_sets,ops):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        # Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("UpPyramid")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
        arr = nan_to_zero(arr,zero=3)
    else:   #CPUs do this
        blocks = []
        for local_region in cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,cpu_regions,ops)
        arr = nan_to_zero(arr,zero=4)
    # Writing to shared array
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

def Octahedron(source_mod,arr,gpu_rank,block_size,grid_size,region,local_cpu_regions,shared_arr,idx_sets,ops,printer=None):
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
        # arr = nan_to_zero(arr,zero=7)
        cuda.Context.synchronize()
        # printer(arr[1,0,:,:])
    else:   #CPUs do this
        blocks = []
        for local_region in local_cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_Octahedron,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,local_cpu_regions,ops)

    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

def DownPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,shared_arr,idx_sets,ops):
    """This is the ending inverted pyramid."""
    if gpu_rank:
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("DownPyramid")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in local_cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_DownPyramid,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,local_cpu_regions,ops)
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

#--------------------------------CPU Specific Swept Functions------------------
def CPU_UpPyramid(args):
    """Use this function to build the Up Pyramid."""
    block,source_mod,idx_sets = args
    #Removing elements for swept step
    ts = 0
    for swept_set in idx_sets:
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    return block

def CPU_Octahedron(args):
    """Use this function to build the Octahedron."""
    block,source_mod,idx_sets = args
    #Removing elements for swept step
    ts = 0
    #Down Pyramid Step
    for swept_set in idx_sets[::-1]:
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    #Up Pyramid Step
    for swept_set in idx_sets[1:]:  #Skips first element because down pyramid
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    return block

def CPU_DownPyramid(args):
    """Use this function to build the Down Pyramid."""
    block,source_mod,idx_sets = args
    #Removing elements for swept step
    ts = 0
    for swept_set in idx_sets[::-1]:
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    return block

def nan_to_zero(arr,zero=0.):
    """Use this function to turn nans to zero."""
    for i in np.ndindex(arr.shape):
        if np.isnan(arr[i]):
            arr[i]=zero
    return arr

def create_shift_regions(regions,SPLITX,SPLITY):
    """Use this function to create shifted regions."""
    shifted_regions = tuple()
    for region in regions:
        temp_region = region[:2]
        temp_region += slice(region[2].start+SPLITX,region[2].stop+SPLITY,1),
        temp_region += slice(region[3].start+SPLITX,region[3].stop+SPLITY,1),
        shifted_regions+=temp_region,
    return shifted_regions

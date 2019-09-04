#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management
# and data decomposition for the swept rule.
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#MPI imports
from mpi4py import MPI
import importlib


def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def create_local_array(shared_arr,region,dType):
    """Use this function to generate the local arrays from regions."""
    local_shape = get_slices_shape(region)
    local_array = np.zeros(local_shape,dtype=dType)
    local_array[:,:,:,:] = shared_arr[region]
    return local_array

def get_affinity_slices(affinity,block_size,arr_shape):
    """Use this function to split the given data based on rank information and the affinity.
    affinity -  a value between zero and one. (GPU work/CPU work)/Total Work
    block_size -  gpu block size
    arr_shape - shape of array initial conditions array (v,x,y)
    ***This function splits to the nearest column, so the affinity may change***
    """
    #Getting number of GPU blocks based on affinity
    blocks_per_column = arr_shape[2]/block_size[1]
    blocks_per_row = arr_shape[1]/block_size[0]
    num_blocks = int(blocks_per_row*blocks_per_column)
    gpu_blocks = round(affinity*num_blocks)
    #Rounding blocks to the nearest column
    col_mod = gpu_blocks%blocks_per_column  #Number of blocks ending in a column
    col_perc = col_mod/blocks_per_column    #What percentage of the column
    gpu_blocks += round(col_perc)*blocks_per_column-col_mod #Round to the closest column and add the appropriate value
    #Getting number of columns and rows
    num_columns = int(gpu_blocks/blocks_per_column)
    #Region Indicies
    gpu_slices = (slice(0,arr_shape[0],1),slice(0,int(block_size[0]*num_columns),1),)
    cpu_slices = (slice(0,arr_shape[0],1),slice(int(block_size[0]*num_columns),arr_shape[1],1),slice(0,arr_shape[2],1))
    return gpu_slices, cpu_slices

def get_slices_shape(slices):
    """Use this function to convert slices into a shape tuple."""
    stuple = tuple()
    for s in slices:
        stuple+=(s.stop-s.start,)
    return stuple

def boundary_update(shared_arr,ops,SPLITX,SPLITY):
    """Use this function to update the boundary point ghost nodes."""
    SX = SPLITX+ops
    SY = SPLITY+ops
    #Back to front
    shared_arr[:,:,:ops,:] = shared_arr[:,:,-SX-ops:-SX,:]
    shared_arr[:,:,:,:ops] = shared_arr[:,:,:,-SY-ops:-SY]
    #Front to back
    shared_arr[:,:,-SX:,:] = shared_arr[:,:,ops:SX+ops,:]
    shared_arr[:,:,:,-SY:] = shared_arr[:,:,:,ops:SY+ops]

def edge_shift(shared_arr,eregions, dir):
    """Use this function to shift edge points>"""
    if dir:
        for bt,bv,x1,y1,x2,y2 in eregions:  #Back to Front
            shared_arr[bt,bv,x1,y1] = shared_arr[bt,bv,x2,y2]
    else:
        for bt,bv,x1,y1,x2,y2 in eregions:  #Front to Back
            shared_arr[bt,bv,x2,y2] = shared_arr[bt,bv,x1,y1]


def hdf_swept_write(cwt,wb,shared_arr,reg,hdf_set,hr,MPSS,TSO):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    for carr in shared_arr[TSO:MPSS+TSO]:
        if (wb+1)%TSO==0:
            hdf_set[cwt,hr[0],hr[1],hr[2]]=carr[reg[1],reg[2],reg[3]]
            cwt+=1
        wb += 1
    #Data is shifted and TSO steps are kept at the begining
    nte = shared_arr.shape[0]-(MPSS)
    shared_arr[:nte,reg[1],reg[2],reg[3]] = shared_arr[MPSS:,reg[1],reg[2],reg[3]]
    shared_arr[nte:,reg[1],reg[2],reg[3]] = 0
    #Do edge comm after this function
    return cwt,wb

#Nan to zero is a testing function - it doesn't really belong here
def nan_to_zero(arr,zero=0.):
    """Use this function to turn nans to zero."""
    for i in np.ndindex(arr.shape):
        if np.isnan(arr[i]):
            arr[i]=zero
    return arr

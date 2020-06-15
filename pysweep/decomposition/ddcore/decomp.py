#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management
# and data decomposition for the swept rule.
import h5py, ctypes,GPUtil,os
import numpy as np
import multiprocessing as mp
from itertools import cycle, product
from mpi4py import MPI
from ddcore import sgs

def create_local_gpu_array(block_shape):
    """Use this function to create the local gpu array"""
    arr = np.zeros(block_shape)
    arr = arr.astype(np.float64)
    arr = np.ascontiguousarray(arr)
    return arr

def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def create_shared_pool_array(shared_shape):
    """Use this function to create the shared array for the process pool."""
    sarr_base = mp.Array(ctypes.c_double, int(np.prod(shared_shape)),lock=False)
    sarr = np.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    return sarr

def read_input_file(comm):
    """This function is to read the input file"""
    file = h5py.File('input_file.hdf5','r',driver='mpio',comm=comm)
    gargs = tuple(file['gargs'])
    swargs = tuple(file['swargs'])
    exid = tuple(file['exid'])
    filename = ''
    for item in file['filename']:
        filename += chr(item)
    dType = ''
    for item in file['dType']:
        dType += chr(item)
    dType = np.dtype(dType)
    arrf = file['arr0']
    arr0 = np.zeros(np.shape(arrf), dtype=dType)
    arr0[:,:,:] = arrf[:,:,:]
    file.close()
    return arr0,gargs,swargs,filename,exid,dType


def create_cpu_blocks(total_cpu_block,BS,shared_shape,ops):
    """Use this function to create blocks."""
    row_range = np.arange(ops,total_cpu_block[2].stop-total_cpu_block[2].start+ops,BS[1],dtype=np.intc)
    column_range = np.arange(ops,total_cpu_block[3].stop-total_cpu_block[3].start+ops,BS[1],dtype=np.intc)
    xslice = slice(shared_shape[2]-2*ops-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    yslice = slice(total_cpu_block[3].start-ops,total_cpu_block[3].stop+ops,1)
    #There is an issue with total cpu block for decomp on cluster
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,yslice)
    return [(total_cpu_block[0],total_cpu_block[1],slice(x-ops,x+BS[0]+ops,1),slice(y-ops,y+BS[1]+ops,1)) for x,y in product(row_range,column_range)],ntcb


def nsplit(rank,node_master,node_comm,num_gpus,node_info,BS,arr_shape,gpu_rank,ops):
    """Use this function to split data amongst nodes."""
    if rank == node_master:
        start,stop,gm,cm = node_info
        gstop = start+gm
        g = np.linspace(start,gstop,num_gpus+1,dtype=np.intc)
        gbs = [slice(int(g[i]*BS[0]+ops),int(g[i+1]*BS[0]+ops),1) for i in range(num_gpus)]+[slice(int(gstop*BS[0]+ops),int(stop*BS[0]+ops),1)]
        gpu_rank = list(zip(gpu_rank,gbs))
    gpu_rank = node_comm.scatter(gpu_rank)
    return gpu_rank

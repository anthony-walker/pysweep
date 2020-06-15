#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management
# and data decomposition for the swept rule.
import h5py, ctypes,GPUtil,os
import numpy as np
import multiprocessing as mp
from itertools import cycle, product
from mpi4py import MPI

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

def get_local_extend_array(sarr,arr,blocks,BS):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    i1,i2,i3,i4 = blocks
    arr[:,:,:,BS[0]:-BS[0]] = sarr[blocks]
    arr[:,:,:,0:BS[0]] = sarr[i1,i2,i3,-BS[0]:]
    arr[:,:,:,-BS[0]:] = sarr[i1,i2,i3,:BS[0]]
    return arr

def create_shared_pool_array(shared_shape):
    """Use this function to create the shared array for the process pool."""
    sarr_base = mp.Array(ctypes.c_double, int(np.prod(shared_shape)),lock=False)
    sarr = np.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    return sarr

def read_input_file(comm,filename = 'input_file.hdf5'):
    """This function is to read the input file"""
    file = h5py.File(filename,'r',driver='mpio',comm=comm)
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

def create_cpu_blocks(total_cpu_block,BS,shared_shape):
    """Use this function to create blocks."""
    row_range = np.arange(0,total_cpu_block[2].stop-total_cpu_block[2].start,BS[1],dtype=np.intc)
    column_range = np.arange(total_cpu_block[3].start,total_cpu_block[3].stop,BS[1],dtype=np.intc)
    xslice = slice(shared_shape[2]-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,total_cpu_block[3])
    return [(total_cpu_block[0],total_cpu_block[1],slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)],ntcb

def create_escpu_blocks(cpu_blocks,shared_shape,BS):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    shift = int(BS[1]/2)
    #Creating shift blocks and edges
    shift_blocks = [(block[0],block[1],block[2],slice(block[3].start+shift,block[3].stop+shift,1)) for block in cpu_blocks]
    for i,block in enumerate(shift_blocks):
        if block[3].start < 0:
            new_block = (block[0],block[0],block[2],np.arange(block[3].start,block[3].stop))
            shift_blocks[i] = new_block
        elif block[3].stop > shared_shape[3]:
            ny = np.concatenate((np.arange(block[3].start,block[3].stop-shift),np.arange(0,shift,1)))
            new_block = (block[0],block[0],block[2],ny)
            shift_blocks[i] = new_block
    return cpu_blocks,shift_blocks

def nsplit(rank,node_master,node_comm,num_gpus,node_info,BS,arr_shape,gpu_rank):
    """Use this function to split data amongst nodes."""
    if rank == node_master:
        start,stop,gm,cm = node_info
        gstop = start+gm
        g = np.linspace(start,gstop,num_gpus+1,dtype=np.intc)
        gbs = [slice(int(g[i]*BS[0]),int(g[i+1]*BS[0]),1) for i in range(num_gpus)]+[slice(int(gstop*BS[0]),int(stop*BS[0]),1)]
        gpu_rank = list(zip(gpu_rank,gbs))
    gpu_rank = node_comm.scatter(gpu_rank)
    return gpu_rank

def swept_write(cwt,sarr,hdf_data,gsc,gts,TSO,MPSS):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    i1,i2,i3=gsc #Unpack global tuple
    for si,i in enumerate(range(gts+1,gts+1+MPSS,1),start=TSO):
        if i%TSO==0:
            hdf_data[cwt,i1,i2,i3] = sarr[si,:,:,:]
            cwt+=1
    nte = sarr.shape[0]-MPSS
    sarr[:nte,:,:,:] = sarr[MPSS:,:,:,:]
    sarr[nte:,:,:,:] = 0
    return cwt

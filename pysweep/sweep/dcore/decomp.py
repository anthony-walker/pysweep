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
    arr = arr.astype(np.float32)
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

def copy_s_to_g(sarr,arr,blocks,BS):
    """Use this function to copy the shared array to the local array."""
    i1,i2,i3,i4 = blocks
    arr[:,:,:,BS[0]:-BS[0]] = sarr[blocks]
    arr[:,:,:,0:BS[0]] = sarr[i1,i2,i3,-BS[0]:]
    arr[:,:,:,-BS[0]:] = sarr[i1,i2,i3,:BS[0]]
    return arr

def create_shared_pool_array(shared_shape):
    """Use this function to create the shared array for the process pool."""
    sarr_base = mp.Array(ctypes.c_float, int(np.prod(shared_shape)),lock=False)
    sarr = np.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    return sarr

# def create_shared_array(sarr):
#    """Use this to create shared memory array, requires 3.8 which has yet to be released"""
#     shm = mp.shared_memory.SharedMemory(create=True,size=sarr.nbytes)
#     return np.ndarray(sarr.shape,dtype=sarr.dtype,buffer=shm.buf)


def create_shared_mpi_array(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def read_input_file(comm):
    """This function is to read the input file"""
    file = h5py.File('input_file.hdf5','r',driver='mpio',comm=comm)
    gargs = tuple(file['gargs'])
    swargs = tuple(file['swargs'])
    exid = tuple(file['exid'])
    GS = ''
    for item in file['GS']:
        GS += chr(item)
    CS = ''
    for item in file['CS']:
        CS += chr(item)
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
    return arr0,gargs,swargs,GS,CS,filename,exid,dType


def create_cpu_blocks(total_cpu_block,BS):
    """Use this function to create blocks."""
    row_range = np.arange(0,total_cpu_block[2].stop-total_cpu_block[2].start,BS[1],dtype=np.intc)
    column_range = np.arange(total_cpu_block[3].start,total_cpu_block[3].stop,BS[1],dtype=np.intc)
    return [(total_cpu_block[0],total_cpu_block[1],slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)]

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
    return list(zip(cpu_blocks,shift_blocks))

def create_es_blocks(cpu_blocks,shared_shape,ops,BS):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    s0 = slice(0,shared_shape[0],1)
    s1 = slice(0,shared_shape[1],1)
    shift = int(BS[1]/2)
    #Creating shift blocks and edges
    shift_blocks = [(s0,s1,block[2],slice(block[3].start+shift,block[3].stop+shift,1)) for block in cpu_blocks]
    for i,block in enumerate(shift_blocks):
        if block[3].start < 0:
            new_block = (s0,s1,block[2],np.arange(block[3].start,block[3].stop))
            shift_blocks[i] = new_block
        elif block[3].stop > shared_shape[3]:
            ny = np.concatenate((np.arange(block[3].start,block[3].stop-shift),np.arange(0,shift,1)))
            new_block = (s0,s1,block[2],ny)
            shift_blocks[i] = new_block

    # #Creating non-shift block edges
    # for i,block in enumerate(cpu_blocks):
    #     if block[3].start < 0:
    #         new_block = (s0,s1,block[2],np.arange(block[3].start,block[3].stop))
    #         cpu_blocks[i] = new_block
    #     elif block[3].stop > shared_shape[3]:
    #         ny = np.concatenate((np.arange(block[3].start,block[3].stop-ops),np.arange(0,ops,1)))
    #         new_block = (s0,s1,block[2],ny)
    #         cpu_blocks[i] = new_block
    return zip(cpu_blocks,shift_blocks)

def create_blocks(shared_shape,rows_per_gpu,BS,num_gpus,ops):
    """Use this function to create blocks."""
    gpu_blocks = []
    s0 = slice(0,shared_shape[0],1)
    s1 = slice(0,shared_shape[1],1)
    for i in range(num_gpus):
        gpu_blocks.append((s0,s1,slice(int(BS[0]*rows_per_gpu*i),int(BS[0]*rows_per_gpu*(i+1)),1),slice(0,shared_shape[3],1)))
    # cstart = gpu_blocks[-1][2].stop if gpu_blocks else 0
    cstart =  0
    gstop = gpu_blocks[-1][2].stop if gpu_blocks else 0
    total_cpu_block = (s0,s1,slice(gstop,shared_shape[2],1),slice(0,shared_shape[3],1))
    row_range = np.arange(cstart,shared_shape[2]-gstop,BS[0],dtype=np.intc)
    column_range = np.arange(0,shared_shape[3],BS[1],dtype=np.intc)
    cpu_blocks = [(s0,s1,slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)]
    return gpu_blocks, cpu_blocks, total_cpu_block


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

def node_split(nodes,node,master_node,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus):
    """Use this function to split data amongst nodes."""
    #Assert that the total number of blocks is an integer
    gpu_rows = np.ceil(num_block_rows*AF)
    rows_per_gpu = int(gpu_rows/total_num_gpus) if total_num_gpus > 0 else 0
    cpu_rows = num_block_rows-gpu_rows

    rows_per_core = np.ceil(cpu_rows/total_num_cores) if num_cores > 0 else 0
    #Estimate number of rows per node
    node_rows = rows_per_gpu*num_gpus+rows_per_core*num_cores
    nlst = None
    #Number of row correction
    tnr = nodes.gather((node,node_rows))
    cores = nodes.gather(num_cores)
    if node == master_node:
        nlst = [x for x,y in tnr]
        tnr = [y for x,y in tnr]
        total_rows = np.sum(tnr)
        cis = cycle(tuple(np.zeros(nodes.Get_size()-1))+(1,))
        ncyc = cycle(range(0,nodes.Get_size(),1))
        #If total rows are greater, reduce
        min_cores = min(cores)
        while total_rows > num_block_rows:
            curr = next(ncyc)
            if min_cores >= cores[curr]:
                tnr[curr]-=1
                total_rows-=1
            min_cores += next(cis)
        #If total rows are greater, add
        max_cores = max(cores)
        while total_rows < num_block_rows:
            curr = next(ncyc)
            if max_cores <= cores[curr]:
                tnr[curr]+=1
                total_rows+=1
            max_cores -= next(cis)
    nodes.Barrier()
    return nodes.bcast((nlst,tnr),root=master_node)+(rows_per_gpu,)

#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management
# and data decomposition for the swept rule.
import numpy as np
from mpi4py import MPI
import multiprocessing as mp
import importlib, h5py
from itertools import cycle, product
import ctypes

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


def node_split(nodes,node,master_node,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus):
    """Use this function to split data amongst nodes."""
    #Assert that the total number of blocks is an integer
    gpu_rows = np.ceil(num_block_rows*AF)
    rows_per_gpu = gpu_rows/total_num_gpus if total_num_gpus > 0 else 0
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



def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def create_local_gpu_array(block_shape):
    """Use this function to create the local gpu array"""
    arr = np.zeros(block_shape)
    arr = arr.astype(np.float32)
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

def create_shared_mpi_array(comm,arr_shape,dType,arr_bytes):
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

def hdf_swept_write(cwt,wb,shared_arr,reg,hdf_set,hr,MPSS,TSO,IEP):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""

    wb+=IEP
    for carr in shared_arr[TSO+IEP:MPSS+TSO]:
        if (wb)%TSO==0:
            hdf_set[cwt,hr[0],hr[1],hr[2]]=carr[reg[1],reg[2],reg[3]]
            cwt+=1
        wb += 1
    #Data is shifted and TSO steps are kept at the begining
    nte = shared_arr.shape[0]+1-(MPSS)
    shared_arr[:nte,reg[1],reg[2],reg[3]] = shared_arr[MPSS-1:,reg[1],reg[2],reg[3]]
    shared_arr[nte:,reg[1],reg[2],reg[3]] = 0
    #Do edge comm after this function

    return cwt,wb+1

#Nan to zero is a testing function - it doesn't really belong here
def nan_to_zero(arr,zero=0.):
    """Use this function to turn nans to zero."""
    for i in np.ndindex(arr.shape):
        if np.isnan(arr[i]):
            arr[i]=zero
    return arr

#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management and data decomposition for the swept rule.

import numpy, mpi4py.MPI as MPI
from itertools import product

def create_local_gpu_array(block_shape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(block_shape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return arr

def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = numpy.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
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
    sarr_base = mp.Array(ctypes.c_double, int(numpy.prod(shared_shape)),lock=False)
    sarr = numpy.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    return sarr

def create_swept_cpu_blocks(total_cpu_block,BS,shared_shape):
    """Use this function to create blocks."""
    row_range = numpy.arange(0,total_cpu_block[2].stop-total_cpu_block[2].start,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(total_cpu_block[3].start,total_cpu_block[3].stop,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,total_cpu_block[3])
    return [(total_cpu_block[0],total_cpu_block[1],slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)],ntcb

def create_standard_cpu_blocks(total_cpu_block,BS,shared_shape,ops):
    """Use this function to create blocks."""
    row_range = numpy.arange(ops,total_cpu_block[2].stop-total_cpu_block[2].start+ops,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(ops,total_cpu_block[3].stop-total_cpu_block[3].start+ops,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-2*ops-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    yslice = slice(total_cpu_block[3].start-ops,total_cpu_block[3].stop+ops,1)
    #There is an issue with total cpu block for decomp on cluster
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,yslice)
    return [(total_cpu_block[0],total_cpu_block[1],slice(x-ops,x+BS[0]+ops,1),slice(y-ops,y+BS[1]+ops,1)) for x,y in product(row_range,column_range)],ntcb

def create_swept_escpu_blocks(cpu_blocks,shared_shape,BS):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    shift = int(BS[1]/2)
    #Creating shift blocks and edges
    shift_blocks = [(block[0],block[1],block[2],slice(block[3].start+shift,block[3].stop+shift,1)) for block in cpu_blocks]
    for i,block in enumerate(shift_blocks):
        if block[3].start < 0:
            new_block = (block[0],block[0],block[2],numpy.arange(block[3].start,block[3].stop))
            shift_blocks[i] = new_block
        elif block[3].stop > shared_shape[3]:
            ny = numpy.concatenate((numpy.arange(block[3].start,block[3].stop-shift),numpy.arange(0,shift,1)))
            new_block = (block[0],block[0],block[2],ny)
            shift_blocks[i] = new_block
    return cpu_blocks,shift_blocks

def nsplit(rank,node_master,node_comm,num_gpus,node_info,BS,arr_shape,gpu_rank,ops=None,swept=True):
    """Use this function to split data amongst nodes."""
    if rank == node_master:
        start,stop,gm,cm = node_info
        gstop = start+gm
        g = numpy.linspace(start,gstop,num_gpus+1,dtype=numpy.intc)
        if swept: #Swept rule
            gbs = [slice(int(g[i]*BS[0]),int(g[i+1]*BS[0]),1) for i in range(num_gpus)]+[slice(int(gstop*BS[0]),int(stop*BS[0]),1)]
        else: #Standard
            gbs = [slice(int(g[i]*BS[0]+ops),int(g[i+1]*BS[0]+ops),1) for i in range(num_gpus)]+[slice(int(gstop*BS[0]+ops),int(stop*BS[0]+ops),1)]
        gpu_rank = list(zip(gpu_rank,gbs))
    gpu_rank = node_comm.scatter(gpu_rank)
    return gpu_rank

def create_dist_up_sets(block_size,ops):
    """This function creates up sets for the dsweep."""
    sets = tuple()
    ul = block_size[0]-ops
    dl = ops
    while ul > dl:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl+=ops
        ul-=ops
    return sets

def create_dist_down_sets(block_size,ops):
    """Use this function to create the down pyramid sets from up sets."""
    bsx = int(block_size[0]/2)
    bsy = int(block_size[1]/2)
    dl = int((bsy)-ops); #lower y
    ul = int((bsy)+ops); #upper y
    sets = tuple()
    while dl > 0:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl-=ops
        ul+=ops
    return sets

def create_dist_bridge_sets(block_size,ops,MPSS):
    """Use this function to create the iidx sets for bridges."""
    sets = tuple()
    xul = block_size[0]-ops
    xdl = ops
    yul = int(block_size[0]/2+ops)
    ydl = int(block_size[0]/2-ops)
    xts = xul
    xbs = xdl
    for i in range(MPSS):
        sets+=(tuple(product(np.arange(xdl,xul,1),np.arange(ydl,yul,1))),)
        xdl+=ops
        xul-=ops
        ydl-=ops
        yul+=ops
    return sets,sets[::-1]

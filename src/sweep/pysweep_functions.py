#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
from pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
#MPI imports
from mpi4py import MPI

def create_map():
    """Use this function to create local maps for process communication."""
    smap = set() #Each rank has its own map


def affinity_split(affinity,block_size,arr_shape,total_gpus):
    """Use this function to split the given data based on rank information and the affinity.
    affinity -  a value between zero and one. (GPU work/CPU work)/Total Work
    block_size -  gpu block size
    arr_shape - shape of array initial conditions array (v,x,y)
    Assuming total number of gpus is the same as the number of cpus
    """
    #Getting number of GPU blocks based on affinity
    blocks_per_row = arr_shape[2]/block_size[1]
    blocks_per_column = arr_shape[1]/block_size[0]
    num_blocks = int(blocks_per_row*blocks_per_column)
    gpu_blocks = round(affinity*num_blocks)
    #Rounding blocks to the nearest column
    col_mod = gpu_blocks%blocks_per_column
    col_perc = col_mod/blocks_per_column
    gpu_blocks += round(col_perc)*blocks_per_column-col_mod
    #Getting number of columns and rows
    num_columns = gpu_blocks/blocks_per_column
    num_rows = blocks_per_row
    #Region Indicies
    gpu_slices = (slice(0,arr_shape[0],1),slice(0,int(block_size[0]*num_columns),1),slice(0,int(block_size[1]*num_rows),1))
    cpu_slices = (slice(0,arr_shape[0],1),slice(int(block_size[0]*num_columns),int(block_size[0]*blocks_per_column),1),slice(0,int(block_size[1]*num_rows),1))
    return gpu_slices, cpu_slices

def region_split(rank,gargs,cargs,block_size):
    """Use this function to separate the CPU data ."""

    region = None   #region
    #Unpack arguments
    gpu_comm,gpu_master_rank,new_gpu_shape,total_gpus,gpu_rank = gargs
    cpu_comm,cpu_master_rank,new_cpu_shape,total_cpus = cargs
    #GPU ranks go in here
    if gpu_rank:
        #Getting gpu blocks
        gpu_blocks = int(new_gpu_shape[3]/block_size[1])
        blocks_per_gpu = int(gpu_blocks/total_gpus)
        #creating gpu indicies
        if gpu_master_rank == rank:
            stuple = tuple()
            #Creating slice prefixes
            for i in range(len(new_gpu_shape)-1):
                stuple+=(slice(0,new_gpu_shape[i],1),)
            #Creating regions
            prev = 0
            region = list()    #a list of slices one for each region
            for i in range(blocks_per_gpu,gpu_blocks+1,blocks_per_gpu):
                region.append(stuple+(slice(prev*block_size[1],i*block_size[1],1),))
                prev = i
        region = gpu_comm.scatter(region)

    #CPU ranks go in here
    else:
        #Getting cpu blocks
        cpu_blocks = int(new_cpu_shape[3]/block_size[1])
        blocks_per_cpu = int(np.ceil(cpu_blocks/total_cpus))   #Ensures all blocks will be given to a cpu core
        #creating cpu indicies
        if cpu_master_rank == rank:
            stuple = tuple()
            #Creating slice prefixes
            for i in range(len(new_cpu_shape)-1):
                stuple+=(slice(0,new_cpu_shape[i],1),)
            #Creating regions
            prev = 0
            region = list()    #a list of slices one for each region
            for i in range(blocks_per_cpu,cpu_blocks+1,blocks_per_cpu):
                region.append(stuple+(slice(prev*block_size[1],i*block_size[1],1),))
                prev = i
            if len(region)<total_cpus:
                for i in range(abs(len(region)-total_cpus)):
                    region.append(stuple+(slice(0,0,1),))
        region = cpu_comm.scatter(region)
    return region

def extended_shape(orig_shape,block_size):
    """Use this function to develop extended array shapes."""
    return (orig_shape[0],orig_shape[1]+int(block_size[0]/2),orig_shape[2])

def get_gpu_ranks(rank_info):
    """Use this funciton to determine which ranks with have a GPU.
        given: rank_info - a list of tuples of (rank, processor, gpu_ids, gpu_rank)
        returns: new_rank_inf, total_gpus
    """
    total_gpus = 0
    new_rank_info = list()
    assigned_ctr = dict()
    devices = dict()
    for ri in rank_info:
        temp = ri
        if ri[1] not in assigned_ctr:
            assigned_ctr[ri[1]] = len(ri[2])
            ri[2].reverse()
            devices[ri[1]] = ri[2].copy()   #Device ids
            total_gpus+=len(ri[2])

        if assigned_ctr[ri[1]] > 0:
            temp = ri[:-2]+(True,)+(devices[ri[1]].pop(),)
            assigned_ctr[ri[1]]-=1
        new_rank_info.append(temp)
    return new_rank_info, total_gpus

def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def source_code_read(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source

def constant_copy(source_mod,ps,grid_size,block_size,ops,add_const=None):
    """Use this function to copy constant args to cuda memory.
        source_mod - the source module obtained from pycuda and source code
        ps - shape of the 4D array (t,v,x,y)
        block_size -  size of blocks to run on GPU
        grid_size -  size of GPU grid
        ops - number of atomic operations
        add_const - additional dictionary with key(global):(cast_fcn, data)
        Note, the key should match the global in the source_mod and the cast_fcn
        should convert the data to the appropriate type.
        e.g.  int_cast = lambda x:np.int32(x) add_const = {SOME_CONST:(int_cast,myInt)}
    """
    #Functions to cast data to appropriate gpu types
    int_cast = lambda x:np.int32(x)
    float_cast = lambda x:np.float32(x)

    #Dictionary for constants
    const_dict = dict()
    #Generating constants
    MSS = min(block_size[2:])/(2*ops) #Max swept step
    NV = ps[1]  #Number of variables in given initial array
    SGIDS = block_size[0]*block_size[1]
    VARS =  SGIDS*grid_size[0]*grid_size[1]
    TIMES = VARS*NV
    const_dict['MSS'] = (int_cast,MSS)
    const_dict['NV'] = (int_cast,NV)
    const_dict['SGIDS'] = (int_cast,SGIDS)
    const_dict['VARS'] = (int_cast,VARS)
    const_dict['TIMES'] = (int_cast,TIMES)
    const_dict['OPS'] = (int_cast,ops)
    const_dict['SGNVS'] = (int_cast,SGIDS*NV)
    for key in const_dict:
        c_ptr,_ = source_mod.get_global(key)
        cuda.memcpy_htod(c_ptr,const_dict[key][0](const_dict[key][1]))
    return MSS,NV,SGIDS,VARS,TIMES,const_dict

def UpPyramid(arr, ops):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    plane_shape = np.shape(arr)
    iidx = list(np.ndindex(plane_shape[2:]))
    #Bounds
    lb = 0
    ub = [plane_shape[2],plane_shape[3]]
    #Going through all swept steps
    # pts = [iidx]    #This is strictly for debugging
    while ub[0] > lb and ub[1] > lb:
        lb += ops
        ub = [x-ops for x in ub]
        iidx = [x for x in iidx if x[0]>=lb and x[1]>=lb and x[0]<ub[0] and x[1]<ub[1]]
        if iidx:
            step(arr,iidx,0)
            # pts.append(iidx) #This is strictly for debuggings
    # return pts #This is strictly for

def Octahedron(arr,  ops):
    """This is the step(s) in between UpPyramid and DownPyramid."""
    pass

def DownPyramid(arr, ops):
    """This is the ending inverted pyramid."""
    pass

#--------------------------------NOT USED CURRENTLY-----------------------------
def edges(arr,ops,shape_adj=-1):
    """Use this function to generate boolean arrays for edge handling."""
    mask = np.zeros(arr.shape[:shape_adj], dtype=bool)
    mask[(arr.ndim+shape_adj)*(slice(ops, -ops),)] = True
    return mask

def dummy_fcn(arr):
    """This is a testing function for arch_speed_comp."""
    iidx = np.ndindex(np.shape(arr))
    for i,idx in enumerate(iidx):
        arr[idx] *= i
        # print(i,value)
    return arr

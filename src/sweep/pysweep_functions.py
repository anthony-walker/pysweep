#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#MPI imports
from mpi4py import MPI
import importlib
#Multiprocessing
import multiprocessing as mp

def local_array_create(shared_arr,region,dType):
    """Use this function to generate the local arrays from regions."""
    if region:
        local_shape = get_slices_shape(region)
        local_array = np.zeros(local_shape,dtype=dType)
        local_array[:,:,:,:] = shared_arr[region]
    else:
        local_array = None

    return local_array

def edge_comm(shared_arr,SPLITX,SPLITY):
    """Use this function to communicate edges in the shared array."""
    shared_arr[:,:,-SPLITX:,:] = shared_arr[:,:,:SPLITX,:]
    shared_arr[:,:,:,-SPLITY:] = shared_arr[:,:,:,:SPLITY]

def build_cpu_source(cpu_source):
    """Use this function to build source module from cpu code."""
    module_name = cpu_source.split("/")[-1]
    module_name = module_name.split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, cpu_source)
    source_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_mod)
    return source_mod

def build_gpu_source(kernel_source):
    """Use this function to build the given and swept source module together.
    -NEED TO UPDATE THIS TO GET PATH OF SWEPT SOURCE IN LIEU OF HARDWIRED
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    source_code = source_code_read("/home/walkanth/pysweep/src/sweep/sweep.h")
    split_source_code = source_code.split("//!!(@#\n")
    source_code = split_source_code[0]+"\n"+source_code_read(kernel_source)+"\n"+split_source_code[1]
    source_mod = SourceModule(source_code)#,options=["--ptxas-options=-v"])
    return source_mod

def get_slices_shape(slices):
    """Use this function to convert slices into a shape tuple."""
    stuple = tuple()
    for s in slices:
        stuple+=(s.stop-s.start,)
    return stuple

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

def region_split(rank,gargs,cargs,block_size,MOSS):
    """Use this function to obtain the regions to for reading and writing
        from the shared array. region 1 is standard region 2 is offset by split.
        Note: The rows are divided into regions. So, each rank gets a row or set of rows
        so to speak. This is because affinity split is based on columns.
    """
    region1 = None   #region1
    region2 = None   #region2
    SPLITX = int(block_size[0]/2)
    SPLITY = int(block_size[1]/2)
    #Unpack arguments
    gpu_comm,gpu_master_rank,gpu_shape,total_gpus,gpu_rank = gargs
    cpu_comm,cpu_master_rank,cpu_shape,total_cpus = cargs
    stv = (slice(0,MOSS,1),slice(0,gpu_shape[0],1))
    #GPU ranks go in here
    if gpu_rank:
        #Getting gpu blocks
        gpu_blocks = int(gpu_shape[2]/block_size[1])
        blocks_per_gpu = int(gpu_blocks/total_gpus)
        #creating gpu indicies
        if gpu_master_rank == rank:
            stx1=(slice(0,gpu_shape[1],1),)
            stx2=(slice(SPLITX,gpu_shape[1]+SPLITX,1),)
            #Creating regions
            prev = 0
            region1 = list()    #a list of slices one for each region
            region2 = list()    #a list of slices one for each region
            for i in range(blocks_per_gpu,gpu_blocks+1,blocks_per_gpu):
                region1.append(stv+stx1+(slice(prev*block_size[1],i*block_size[1],1),))
                region2.append(stv+stx2+(slice(prev*block_size[1]+SPLITY,i*block_size[1]+SPLITY,1),))
                prev = i
        region1 = gpu_comm.scatter(region1)
        region2 = gpu_comm.scatter(region2)
    #CPU ranks go in here
    else:
        #Getting cpu blocks
        cpu_blocks = int(cpu_shape[2]/block_size[1])
        blocks_per_cpu = int(np.ceil(cpu_blocks/total_cpus))   #Ensures all blocks will be given to a cpu core
        #creating cpu indicies
        if cpu_master_rank == rank:
            stuple1 = tuple()
            stuple2 = tuple()
            #Creating slice prefixes
            stx1=(slice(0,cpu_shape[1],1),)
            stx2=(slice(SPLITX,cpu_shape[1]+SPLITX,1),)
            #Creating regions
            prev = 0
            region1 = list()    #a list of slices one for each region
            region2 = list()    #a list of slices one for each region
            if stx1[0].stop>0:
                for i in range(blocks_per_cpu,cpu_blocks+1,blocks_per_cpu):
                    region1.append(stv+stx1+(slice(prev*block_size[1],i*block_size[1],1),))
                    region2.append(stv+stx2+(slice(prev*block_size[1]+SPLITY,i*block_size[1]+SPLITY,1),))
                    prev = i
            else:
                for i in range(blocks_per_cpu,cpu_blocks+1,blocks_per_cpu):
                    region1.append(tuple())
                    region2.append(tuple())
                    prev = i
            if len(region1)<total_cpus: #Appending empty tuples if there are to many processes
                for i in range(abs(len(region1)-total_cpus)):
                    region1.append(tuple())
                    region2.append(tuple())
        region1 = cpu_comm.scatter(region1)
        region2 = cpu_comm.scatter(region2)
    return region1, region2

def get_gpu_ranks(rank_info,affinity):
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
        if assigned_ctr[ri[1]] > 0 and affinity > 0:
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

def constant_copy(source_mod,const_dict,add_const=None):
    """Use this function to copy constant args to cuda memory.
        source_mod - the source module obtained from pycuda and source code
        const_dict - dictionary of constants where the key is the global
        add_const - additional dictionary with constants. Note, they should have
                    the correct type.
    """
    #Functions to cast data to appropriate gpu types
    int_cast = lambda x:np.int32(x)
    float_cast = lambda x:np.float32(x)
    casters = {type(0.1):float_cast,type(1):int_cast}
    #Generating constants
    for key in const_dict:
        c_ptr,_ = source_mod.get_global(key)
        cst = const_dict[key]
        cuda.memcpy_htod(c_ptr,casters[type(cst)](cst))

    for key in add_const:
        c_ptr,_ = source_mod.get_global(key)
        cuda.memcpy_htod(c_ptr,add_const[key])

def UpPyramid(source_mod,arr,ops,gpu_rank,block_size,grid_size,region,shared_arr):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        #Getting GPU Function
        gpu_fcn = source_mod.get_function("UpPyramid")
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=arr[0,:,:block_size[0],:block_size[1]].nbytes)
        cuda.Context.synchronize()
        # for row in arr[1,0,:,:]:
        #     print(row)
    else:   #CPUs do this
        bsx = int(arr.shape[2]/block_size[0])
        bsy =  int(arr.shape[3]/block_size[1])
        cpu_blocks = bsx*bsy
        hcs = lambda x,s,a: np.array_split(x,s,axis=a)   #Lambda function for creating blocks for cpu testing
        blocks =  [item for subarr in hcs(arr,bsx,2) for item in hcs(subarr,bsy,3)]    #applying lambda function
        cpu_fcn = sweep_lambda((CPU_UpPyramid,source_mod,ops))
        blocks = list(map(cpu_fcn,blocks))
        #Rebuild blocks into array
        if len(blocks)>1:
            temp = []
            for i in range(bsy,len(blocks)+1,bsy):
                temp.append(np.concatenate(blocks[i-bsy:i],axis=3))
            if len(temp)>1:
                arr = np.concatenate(temp,axis=2)
            else:
                arr = temp[0]
    shared_arr[region] = arr[:,:,:,:]


def Octahedron(arr,  ops):
    """This is the step(s) in between UpPyramid and DownPyramid."""
    pass

def DownPyramid(arr, ops):
    """This is the ending inverted pyramid."""
    pass
#--------------------------------CPU Specific Swept Functions------------------

def CPU_UpPyramid(args):
    """Use this function to solve a block on the CPU."""
    block,source_mod,ops = args
    plane_shape = block.shape
    bsx = block.shape[2]
    bsy = block.shape[3]
    iidx = tuple(np.ndindex(plane_shape[2:]))
    #Removing elements for swept step
    ts = 0
    while len(iidx)>0:
        #Adjusting indices
        tl = tuple()
        iidx = iidx[ops*bsx:-ops*bsx]
        bsy-=2*ops
        for i in range(1,bsy+1,1):
            tl+=iidx[i*bsx-bsx+ops:i*bsx-ops]
        bsx-=2*ops
        iidx = tl
        #Calculating Step
        block = source_mod.step(block,iidx,ts)
        ts+=1
    return block

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
def create_map():
    """Use this function to create local maps for process communication."""
    smap = set() #Each rank has its own map

def extended_shape(orig_shape,block_size):
    """Use this function to develop extended array shapes."""
    return (orig_shape[0],orig_shape[1]+int(block_size[0]/2),orig_shape[2])

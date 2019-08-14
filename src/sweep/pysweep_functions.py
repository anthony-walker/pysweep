#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
import numpy as np
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#MPI imports
from mpi4py import MPI
import importlib
#System imports
import os.path as op
import inspect

#Multiprocessing
import multiprocessing as mp

def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def affinity_split(affinity,block_size,arr_shape):
    """Use this function to split the given data based on rank information and the affinity.
    affinity -  a value between zero and one. (GPU work/CPU work)/Total Work
    block_size -  gpu block size
    arr_shape - shape of array initial conditions array (v,x,y)
    ***This function splits to the nearest column, so the affinity may change***
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

def boundary_update(shared_arr,ops):
    """Use this function to update the boundary point ghost nodes."""
    shared_arr[0,:,:ops,:] = shared_arr[0,:,-2*ops:-ops,:]
    shared_arr[0,:,:,:ops] = shared_arr[0,:,:,-2*ops:-ops]
    shared_arr[0,:,-ops:,:] = shared_arr[0,:,ops:2*ops,:]
    shared_arr[0,:,:,-ops:] = shared_arr[0,:,:,ops:2*ops]

def create_write_region(comm,rank,master,total_ranks,block_size,arr_shape,slices,time_steps,ops):
    """Use this function to split regions amongst the architecture."""
    y_blocks = arr_shape[2]/block_size[1]
    blocks_per_rank = round(y_blocks/total_ranks)
    rank_blocks = (blocks_per_rank*(rank),blocks_per_rank*(rank+1))
    rank_blocks = comm.gather(rank_blocks,root=master)
    if rank == master:
        rem = int(y_blocks-rank_blocks[-1][1])
        if rem > 0:
            ct = 0
            for i in range(rem):
                rank_blocks[i] = (rank_blocks[i][0]+ct,rank_blocks[i][1]+1+ct)
                ct +=1

            for j in range(rem,total_ranks):
                rank_blocks[j] = (rank_blocks[j][0]+ct,rank_blocks[j][1]+ct)
        elif rem < 0:
            ct = 0
            for i in range(rem,0,1):
                rank_blocks[i] = (rank_blocks[i][0]+ct,rank_blocks[i][1]-1+ct)
                ct -=1
    rank_blocks = comm.scatter(rank_blocks,root=master)
    x_slice = slice(slices[1].start+ops,slices[1].stop+ops,1)
    y_slice = slice(int(block_size[1]*rank_blocks[0]+ops),int(block_size[1]*rank_blocks[1]+ops),1)
    return (slice(0,time_steps,1),slices[0],x_slice,y_slice)

def create_read_region(region,ops):
    """Use this function to obtain the regions to for reading and writing
        from the shared array. region 1 is standard region 2 is offset by split.
        Note: The rows are divided into regions. So, each rank gets a row or set of rows
        so to speak. This is because affinity split is based on columns.
    """
    #Read Region
    new_region = region[:2]
    new_region += slice(region[2].start-ops,region[2].stop+ops,1),
    new_region += slice(region[3].start-ops,region[3].stop+ops,1),
    return new_region

def get_slices_shape(slices):
    """Use this function to convert slices into a shape tuple."""
    stuple = tuple()
    for s in slices:
        stuple+=(s.stop-s.start,)
    return stuple

def create_local_array(shared_arr,region,dType):
    """Use this function to generate the local arrays from regions."""
    local_shape = get_slices_shape(region)
    local_array = np.zeros(local_shape,dtype=dType)
    local_array[:,:,:,:] = shared_arr[region]
    return local_array

def edge_shift(shared_arr,SPLITX,SPLITY,ops,dir):
    """Use this function to communicate edges in the shared array."""
    #Updates shifted section of shared array
    ss = shared_arr.shape
    x_arr = np.zeros(ss[0],ss[1],block_size[0],ss[3])
    y_arr = np.zeros(ss[0],ss[1],ss[2],block_size[1])
    if not dir:
        x_arr[:,:,:,:] = shared_arr[:,:,-block_size[0]-ops:-ops,:]
        shared_arr[:,:,-block_size[0]-ops:-ops,:] = shared_arr[:,:,ops:block_size[0]+ops,:]
        shared_arr[:,:,ops:block_size[0]+ops,:] = x_arr[:,:,:,:]
        #Repeat for y
    else:
        shared_arr[:,:,ops:block_size[0]+2*ops,:]=shared_arr[:,:,-block_size[0]-ops:,:]
        shared_arr[:,:,:,ops:block_size[1]+2*ops]=shared_arr[:,:,:,-block_size[1]-ops:]
    #Updates ops points at front
    shared_arr[:,:,:ops,:] = shared_arr[:,:,-SPLITX-2*ops:-SPLITX-ops,:]
    shared_arr[:,:,:,:ops] = shared_arr[:,:,:,-SPLITY-2*ops:-SPLITY-ops]




def write_and_shift(shared_arr,region1,hdf_set,ops,MPSS,GST):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    r2 = slice(region1[2].start-ops,region1[2].stop-ops,1)
    r3 = slice(region1[3].start-ops,region1[3].stop-ops,1)
    hdf_set[MPSS*(GST-1):MPSS*(GST),region1[1],r2,r3] = shared_arr[:MPSS,region1[1],region1[2],region1[3]]
    shared_arr[:MPSS+1,region1[1],region1[2],region1[3]] = shared_arr[MPSS+1:,region1[1],region1[2],region1[3]]
    #Do edge comm after this function

def create_iidx_sets(block_size,ops):
    """Use this function to create index sets."""
    bsx = block_size[0]+2*ops
    bsy = block_size[1]+2*ops
    ly = ops
    uy = bsy-ops
    min_bs = int(min(bsx,bsy)/(2*ops))
    iidx = tuple(np.ndindex((bsx,bsy)))
    idx_sets = tuple()
    for i in range(min_bs):
        iidx = iidx[ops*(bsy-i*2*ops):-ops*(bsy-i*2*ops)]
        iidx = [(x,y) for x,y in iidx if y >= ly and y < uy]
        if len(iidx)>0:
            idx_sets+=(iidx,)
        ly+=ops
        uy-=ops
    return idx_sets

def local_array_create(shared_arr,region,dType):
    """Use this function to generate the local arrays from regions."""
    if region:
        local_shape = get_slices_shape(region)
        local_array = np.zeros(local_shape,dtype=dType)
        local_array[:,:,:,:] = shared_arr[region]
    else:
        local_array = None
    return local_array

def edge_comm(shared_arr,SPLITX,SPLITY,ops,dir):
    """Use this function to communicate edges in the shared array."""
    #Updates shifted section of shared array
    if not dir:
        shared_arr[:,:,-SPLITX-ops:,:] = shared_arr[:,:,ops:SPLITX+2*ops,:]
        shared_arr[:,:,:,-SPLITY-ops:] = shared_arr[:,:,:,ops:SPLITY+2*ops]
    else:
        shared_arr[:,:,ops:SPLITX+2*ops,:]=shared_arr[:,:,-SPLITX-ops:,:]
        shared_arr[:,:,:,ops:SPLITY+2*ops]=shared_arr[:,:,:,-SPLITY-ops:]
    #Updates ops points at front
    shared_arr[:,:,:ops,:] = shared_arr[:,:,-SPLITX-2*ops:-SPLITX-ops,:]
    shared_arr[:,:,:,:ops] = shared_arr[:,:,:,-SPLITY-2*ops:-SPLITY-ops]

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
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    file = inspect.getfile(build_cpu_source)
    fname = file.split("/")[-1]
    fpath = op.abspath(inspect.getabsfile(build_cpu_source))[:-len(fname)]+"sweep.h"
    source_code = source_code_read(fpath)
    split_source_code = source_code.split("//!!(@#\n")
    source_code = split_source_code[0]+"\n"+source_code_read(kernel_source)+"\n"+split_source_code[1]
    source_mod = SourceModule(source_code)#,options=["--ptxas-options=-v"])
    return source_mod

# def get_slices_shape(slices):
#     """Use this function to convert slices into a shape tuple."""
#     stuple = tuple()
#     for s in slices:
#         stuple+=(s.stop-s.start,)
#     return stuple
#




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

def create_blocks_list(arr,block_size,ops):
    """Use this function to create a list of blocks from the array."""
    if arr is not None:
        bsx = int((arr.shape[2]-2*ops)/block_size[0])
        bsy =  int((arr.shape[3]-2*ops)/block_size[1])
        slices = []
        c_slice = (slice(0,arr.shape[0],1),slice(0,arr.shape[1],1),)
        for i in range(ops+block_size[0],arr.shape[2],block_size[0]):
            for j in range(ops+block_size[1],arr.shape[3],block_size[1]):
                t_slice = c_slice+(slice(i-block_size[0]-ops,i+ops,1),slice(j-block_size[1]-ops,j+ops,1))
                slices.append(t_slice)
        return slices
    else:
        return None

def rebuild_blocks(arr,blocks,local_regions,ops):
    """Use this funciton to rebuild the blocks."""
    #Rebuild blocks into array
    if len(blocks)>1:
        for ct,lr in enumerate(local_regions):
            lr2 = slice(lr[2].start+ops,lr[2].stop-ops,1)
            lr3 = slice(lr[3].start+ops,lr[3].stop-ops,1)
            arr[:,:,lr2,lr3] = blocks[ct][:,:,ops:-ops,ops:-ops]
        return arr
    else:
        return blocks[0]

def UpPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,local_cpu_regions,shared_arr,idx_sets,ops):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        #Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("UpPyramid")
        ss = np.zeros(arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].shape)
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=ss.nbytes)
        cuda.Context.synchronize()
        # arr = nan_to_zero(arr,zero=3)
    else:   #CPUs do this
        blocks = []
        for local_region in local_cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,local_cpu_regions,ops)
        # arr = nan_to_zero(arr,zero=4)
    #Writing to shared array
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

def DownPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,local_cpu_regions,shared_arr,idx_sets,ops):
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

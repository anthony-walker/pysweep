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

def write_and_shift(shared_arr,region1,hdf_set,MPSS):
    """Use this function to write to the hdf file and shift the shared array
        data after writing."""
    hdf_set[MPSS:,region1[1],region1[2],region1[3]] = shared_arr[MPSS:,region1[1],region1[2],region1[3]]
    shared_arr[:MPSS,region1[1],region1[2],region1[3]] = shared_arr[MPSS:,region1[1],region1[2],region1[3]]
    #Do edge comm after this function

def create_iidx_sets(block_size,ops):
    """Use this function to create index sets"""
    b_shape_x = block_size[0]+2*ops
    b_shape_y = block_size[1]+2*ops
    idx_sets = tuple()
    iidx = tuple(np.ndindex((b_shape_x,b_shape_y)))
    # idx_sets += (set(iidx),)
    while len(iidx)>0:
        #Adjusting indices
        tl = tuple()
        iidx = iidx[ops*b_shape_x:-ops*b_shape_x]
        b_shape_y-=2*ops
        for i in range(1,b_shape_y+1,1):
            tl+=iidx[i*b_shape_x-b_shape_x+ops:i*b_shape_x-ops]
        b_shape_x-=2*ops
        iidx = tl
        if len(iidx)>0:
            idx_sets += (set(iidx),)
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

def affinity_split(affinity,block_size,arr_shape,total_gpus,printer=None):
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

def create_write_regions(rank,gargs,cargs,block_size,ops,MOSS,SPLITX,SPLITY,printer=None):
    """Use this function to obtain the regions to for reading and writing
        from the shared array. region 1 is standard region 2 is offset by split.
        Note: The rows are divided into regions. So, each rank gets a row or set of rows
        so to speak. This is because affinity split is based on columns.
    """
    #Read Regions
    region1 = None   #region1
    region2 = None   #region2
    #Write Regions
    region3 = None  #region3
    region4 = None  #region4

    #Unpack arguments
    gpu_comm,gpu_master_rank,total_gpus,gpu_slices,gpu_rank = gargs
    cpu_comm,cpu_master_rank,total_cpus,cpu_slices = cargs
    stv = (slice(0,MOSS,1),gpu_slices[0])
    #GPU ranks go in here
    if gpu_rank:
        #Getting gpu blocks
        gpu_blocks = int((gpu_slices[2].stop-gpu_slices[2].start)/block_size[1])
        blocks_per_gpu = int(gpu_blocks/total_gpus)
        #creating gpu indicies
        if gpu_master_rank == rank:
            x_slice = (slice(gpu_slices[1].start+ops,gpu_slices[1].stop+ops,1),)
            shift_slice=(slice(gpu_slices[1].start+SPLITX+ops,gpu_slices[1].stop+SPLITX+ops,1),)
            #Creating regions
            prev = 0
            region1 = list()    #a list of slices one for each region
            region2 = list()    #a list of slices one for each region
            for i in range(blocks_per_gpu,gpu_blocks+1,blocks_per_gpu):
                region1.append(stv+x_slice+(slice(prev*(block_size[1])+ops,i*(block_size[1])+ops,1),))
                region2.append(stv+shift_slice+(slice(prev*(block_size[1])+SPLITY+ops,i*block_size[1]+SPLITY+ops,1),))
                prev = i
        region1 = gpu_comm.scatter(region1)
        region2 = gpu_comm.scatter(region2)
    #CPU ranks go in here
    else:
        #Getting cpu blocks
        cpu_blocks = int((cpu_slices[2].stop-cpu_slices[2].start)/block_size[1])
        blocks_per_cpu = int(np.ceil(cpu_blocks/total_cpus))   #Ensures all blocks will be given to a cpu core
        #creating cpu indicies
        if cpu_master_rank == rank:
            #Creating slice prefixes
            x_slice = (slice(cpu_slices[1].start+ops,cpu_slices[1].stop+ops,1),)
            shift_slice=(slice(cpu_slices[1].start+SPLITX+ops,cpu_slices[1].stop+SPLITX+ops,1),)
            #Creating regions
            prev = 0
            region1 = list()    #a list of slices one for each region
            region2 = list()    #a list of slices one for each region
            if cpu_slices[0].stop>0 and cpu_slices[1].start != cpu_slices[1].stop:
                for i in range(blocks_per_cpu,cpu_blocks+1,blocks_per_cpu):
                    region1.append(stv+x_slice+(slice(prev*block_size[1]+ops,i*block_size[1]+ops,1),))
                    region2.append(stv+shift_slice+(slice(prev*block_size[1]+SPLITY+ops,i*block_size[1]+SPLITY+ops,1),))
                    prev = i
            for i in range(abs(len(region1)-total_cpus)):
                region1.append(None)
                region2.append(None)
        region1 = cpu_comm.scatter(region1)
        region2 = cpu_comm.scatter(region2)
    return region1, region2

def create_read_regions(regions,ops,printer=None):
    """Use this function to obtain the regions to for reading and writing
        from the shared array. region 1 is standard region 2 is offset by split.
        Note: The rows are divided into regions. So, each rank gets a row or set of rows
        so to speak. This is because affinity split is based on columns.
    """
    if regions[0] is not None:
        id1 = 0
        id2 = 1
        region1 = (regions[id1][0],regions[id1][1])
        region1 += slice(regions[id1][2].start-ops,regions[id1][2].stop+ops,1),
        region1 += slice(regions[id1][3].start-ops,regions[id1][3].stop+ops,1),
        region2 = (regions[id2][0],regions[1][1])
        region2 += slice(regions[id2][2].start-ops,regions[id2][2].stop+ops,1),
        region2 += slice(regions[id2][3].start-ops,regions[id2][3].stop+ops,1),
    else:
        region1 = None
        region2 = None
    return region1,region2

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

def create_blocks_list(arr,block_size,ops,printer=None):
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

def UpPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,local_cpu_regions,shared_arr,idx_sets,ops,printer=None):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        #Getting GPU Function
        printer(arr.shape)
        printer(grid_size)
        # printer("--------------------------------------------")
        # printer(arr[0,0,:,:])
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("UpPyramid")
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=arr[0,:,:block_size[0]+2*ops,:block_size[1]+2*ops].nbytes)
        cuda.Context.synchronize()
        # printer("--------------------------------------------")
        # arr = nan_to_zero(arr)
        # printer(arr[1,0,:,:])
    else:   #CPUs do this
        blocks = []
        for local_region in local_cpu_regions:
            block = np.zeros(arr[local_region].shape)
            block += arr[local_region]
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,local_cpu_regions,ops)
    #Writing to shared array
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

def Octahedron(source_mod,arr,gpu_rank,block_size,grid_size,region,shared_arr,idx_sets):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    if gpu_rank:
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("Octahedron")
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=arr[0,:,:block_size[0],:block_size[1]].nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        bsx = int(arr.shape[2]/block_size[0])
        bsy =  int(arr.shape[3]/block_size[1])
        blocks = [arr[local_region] for local_region in local_cpu_regions]
        cpu_fcn = sweep_lambda((CPU_Octahedron,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(blocks,bsx,bsy)
    shared_arr[region] = arr[:,:,ops:-ops,ops:-ops]

def DownPyramid(source_mod,arr,gpu_rank,block_size,grid_size,region,shared_arr,idx_sets):
    """This is the ending inverted pyramid."""
    if gpu_rank:
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = source_mod.get_function("UpPyramid")
        gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=arr[0,:,:block_size[0],:block_size[1]].nbytes)
        cuda.Context.synchronize()
    else:   #CPUs do this
        bsx = int(arr.shape[2]/block_size[0])
        bsy =  int(arr.shape[3]/block_size[1])
        blocks = create_blocks(arr,bsx,bsy)
        cpu_fcn = sweep_lambda((CPU_DownPyramid,source_mod,idx_sets))
        blocks = list(map(cpu_fcn,blocks))
        # arr = rebuild_blocks(blocks,bsx,bsy)
    shared_arr[region] = arr[:,:,:,:]

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
    for swept_set in idx_sets[::-1][:-1]:
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    #Up Pyramid Step
    for swept_set in idx_sets[2:]:  #Skips first two elements because down pyramid
        #Calculating Step
        block = source_mod.step(block,swept_set,ts)
        ts+=1
    return block

def CPU_DownPyramid(args):
    """Use this function to build the Down Pyramid."""
    block,source_mod,idx_sets = args
    #Removing elements for swept step
    ts = 0
    for swept_set in idx_sets[::-1][:-1]:
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

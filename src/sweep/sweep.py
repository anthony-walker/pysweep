
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import os
import sys

#DataStructure and Math imports
import math
import numpy as np
from collections import deque


#GPU Imports
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except Exception as e:
    print(e)
import GPUtil
from mpi4py import MPI
#Multiprocessing Imports
import multiprocessing as mp
import ctypes

#Testing imports
import platform
import time

def getDeviceAttrs(devNum=0,print_device = False):
    """Use this function to get device attributes and print them"""
    device = cuda.Device(devNum)
    dev_name = device.name()
    dev_pci_bus_id = device.pci_bus_id()
    dev_attrs = device.get_attributes()
    dev_attrs["DEVICE_NAME"]=dev_name
    if print_device:
        for x in dev_attrs:
            print(x,": ",dev_attrs[x])
    return dev_attrs

#--------------------Global Variables------------------------------#
#------------------Shared Memory Arrays-----------------------------------------
cpu_array = None
gpu_array = None
#------------------Architecture-----------------------------------------
# gpu_id = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10) #getting devices by load
# dev_attrs = getDeviceAttrs(gpu_id[0])   #Getting device attributes
# gpu = cuda.Device(gpu_id[0])    #Getting device with id via cuda
# cores = mp.cpu_count()  #Getting number of cpu cores
#----------------------End Global Variables------------------------#

def sweep(y0,ops,block_size, cpu_fcn, gpu_fcn ,gpu_aff=None):
    """Use this function to perform swept rule
    args:
    y0 -  2D numpy array of initial conditions
    ops - number of atomic operations
    block_size - gpu block size (check architecture requirements)
    cpu_fcn - step function to execute swept cpu process (see cpu fcn guidelines)
    gpu_fcn - step function to execute swept cpu process (see cpu fcn guidelines)

    """
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    print("Rank: ",rank," Platform: ",platform.uname()[1])

    #---------------Data Input setup -------------------------#
    plane_shape = np.shape(y0)  #Shape of initial conditions
    max_swept_step = min(block_size[:-1])/(2*ops)

    #----------------Reading In Source Code-------------------------------#
    #Try catch is for working at home
    try:
        source_code = source_code_read("./csrc/swept.h")
        source_code += "\n"+source_code_read("./csrc/euler2D.h")
    except:
        source_code = source_code_read("./pysweep/csrc/swept.h")
        source_code += "\n"+source_code_read("./pysweep/csrc/euler2D.h")
    mod = SourceModule(source_code)


    #------------------Dividing work based on the architecture----------------#
    if rank == master_rank:
        y0s = rank_split(y0,plane_shape,num_ranks)
        if gpu_aff is None:
            arch_speed_comp(mod,dummy_fcn,block_size)
    #----------------------------CUDA ------------------------------#

    #-------------------------END CUDA ---------------------------#

# (gpu_sum,cpu_sum,gpu_id) = arch_query()    #This gets architecture information from ranks
# gpu_affinity *= gpu_sum/cpu_sum #Adjust affinity by number of gpu/cpu_cores
# print(gpu_affinity)
# gpu_blocks,cpu_blocks = arch_work_blocks(plane_shape,block_size,gpu_affinity)
# print(gpu_blocks,cpu_blocks)
# print("Max: ",max_swept_step)

def create_shared_arrays():
    """Use this function to create shared memory arrays for node communication."""
    #----------------------------Creating shared arrays-------------------------#
    global cpu_array
    cpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    cpu_array = np.ctypeslib.as_array(cpu_array_base.get_obj())
    cpu_array = cpu_array.reshape(block_size)

    global gpu_array
    gpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    gpu_array = np.ctypeslib.as_array(gpu_array_base.get_obj())
    gpu_array = gpu_array.reshape(block_size)

def arch_speed_comp(arr,source_mod,cpu_fcn,block_size):
    """This function compares the speed of a block calculation to determine the affinity."""
    num_tries = 10 #Number of tries to get performance ratio
    pool = mp.Pool(cores)   #Pool allocation

    #Creating test blocks
    cpu_test_blocks = list()
    for i in range(cores):
        cpu_test_blocks.append((np.ones(block_size)*2,i))
        # print(cpu_test_blocks[i][0])

    #------------------------Testing CPU Performance---------------------------#
    cpu_performance = 0
    for i in range(num_tries):
        start_cpu = time.time()
        cpu_res = pool.map_async(cpu_fcn,cpu_test_blocks)
        new_cpu_blocks = cpu_res.get()
        stop_cpu = time.time()
        cpu_performance += len(np.prod(block_size)*cpu_test_blocks)/(stop_cpu-start_cpu)
    pool.close()
    pool.join()
    cpu_performance /= num_tries
    print("Average CPU Performance:", cpu_performance)
    #-------------------------Ending CPU Performance Testing--------------------#

    #------------------------Testing GPU Performance---------------------------#
    #Array
    gpu_test_blocks =  np.ones(block_size,dtype=np.float64)*2
    # print(gpu_test_blocks)
    print(type(block_size))
    start_gpu = cuda.Event()
    stop_gpu = cuda.Event()
    start_gpu.record()
    gpu_dummy_fcn = source_mod.get_function("dummy_fcn")
    gpu_dummy_fcn(cuda.InOut(gpu_test_blocks),grid=grid_size,block=block_size)
    stop_gpu.record()
    stop_gpu.synchronize()
    gpu_performance = np.prod(grid_size)*np.prod(block_size)/(start_gpu.time_till(stop_gpu)*1e-3 )
    # print(gpu_test_blocks)
    performance_ratio = gpu_performance/cpu_performance
    print(performance_ratio)
    #-------------------------Ending CPU Performance Testing--------------------#

def dummy_fcn(args):
    """This is a testing function for arch_speed_comp."""
    block = args[0]
    bID = args[1]
    for x in block:
        for y in x:
            y *= y
    return (block,bID)


def archs_phase_1(block_size,num_cpu,num_gpu):
    """Use this function to determine the array splits for the first phase (grid1)"""
    #Axes
    x_ax = 1
    y_ax = 2
    #Getting shape of grid
    grid_shape = np.shape(grid)
    #Creating work element partition integers
    par_x = int(grid_shape[x_ax]/block_size[x_ax])
    par_y = int(grid_shape[y_ax]/block_size[y_ax])
    #Split in y
        #Split in x
            #Add to list

def archs_phase_2(block_size,num_cpu,num_gpu):
    """Use this function to determine the array splits for the second phase (grid2)"""
    #Axes
    x_ax = 1
    y_ax = 2
    #Getting shape of grid
    grid_shape = np.shape(grid)
    #Creating work element partition integers
    par_x = int(grid_shape[x_ax]/block_size[x_ax])
    par_y = int(grid_shape[y_ax]/block_size[y_ax])
    # print("Par(x,y,t): ",par_x,", ",par_y,", ",par_t) #Printing partitions

### ----------------------------- COMPLETED FUNCTIONS -----------------------###
def arch_work_blocks(plane_shape,block_size,gpu_affinity):
    """Use to determine the number of blocks for gpus vs cpus."""
    blocks_x = int(plane_shape[0]/block_size[0])
    blocks_y = int(plane_shape[1]/block_size[1])
    total_blocks = blocks_x*blocks_y
    gpu_blocks = round(total_blocks/(1+1/gpu_affinity))
    block_mod_y = gpu_blocks%blocks_y
    if  block_mod_y!= 0:
        gpu_blocks+=blocks_y-block_mod_y
    cpu_blocks = total_blocks-gpu_blocks
    return (gpu_blocks,cpu_blocks)

def rank_split(y0,plane_shape,rank_size):
    """Use this function to equally split data along the largest axis for ranks."""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(y0,rank_size,axis=major_axis)

def arch_query():
    """Use this method to query the system for information about its hardware"""
    #-----------------------------MPI setup--------------------------------
    comm = MPI.COMM_WORLD
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    rank_info = None    #Set to none for later gather
    cores = mp.cpu_count()  #getting cores of each rank with multiprocessing package

                            #REMOVE ME AFTER TESTING
    #####################################################################
    #Subtracting cores for those used by virtual cluster.
    # if rank == master_rank:
    #     cores -= 14
    #####################################################################

    #Getting avaliable GPUs with GPUtil
    gpu_ids = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
    gpus = 0
    gpu_id = None
    if gpu_ids:
        gpus = len(gpu_ids)
        gpu_id = gpu_ids[0]
    #Gathering all of the information to the master rank
    rank_info = comm.gather((rank, cores, gpus,gpu_id),root=0)
    if rank == master_rank:
        gpu_sum = 0
        cpu_sum = 0
        #Getting total number of cpu's and gpu
        for ri in rank_info:
            cpu_sum += ri[1]
            gpu_sum += ri[2]
            if ri[-1] is not None:
                gpu_id = ri[3]
        return gpu_sum, cpu_sum, gpu_id
    return None,None,gpu_id


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

#NOT USED CURRENTLY
def edges(arr,ops,shape_adj=-1):
    """Use this function to generate boolean arrays for edge handling."""
    mask = np.zeros(arr.shape[:shape_adj], dtype=bool)
    mask[(arr.ndim+shape_adj)*(slice(ops, -ops),)] = True
    return mask

#Swept time space decomposition CPU functions
def UpPyramid(arr, cpu_fcn, ts, ops=2):
    """This is the starting pyramid."""
    plane_shape = np.shape(arr)
    iidx = np.ndindex(plane_shape[:-2])
    #Bounds
    lb = 0
    ub = [plane_shape[0],plane_shape[1]]
    #Going through all swept steps
    while ub[0] > lb and ub[1] > lb:
        lb += ops
        ub = [x-ops for x in ub]
        iidx = [x for x in iidx if x[0]>=lb and x[1]>=lb and x[0]<ub[0] and x[1]<ub[1]]
        cpu_fcn(arr,iidx)


def Octahedron():
    """This is the steps in between UpPyramid and DownPyramid."""
    pass

def DownPyramid():
    """This is the ending inverted pyramid."""
    pass



if __name__ == "__main__":
    # print("Starting execution.")
    dims = (int(20*256),int(5*256),4)
    dims_test = (10,10,5,5)

    y0 = np.zeros(dims_test)
    dy = [0.1,0.1]
    t0 = 0
    t_b = 1
    dt = 0.001
    order = 2
    block_size = (32,32,1)
    GA = 40
    UpPyramid(y0,0,0)
    # sweep(y0,dy,t0,t_b,dt,2,block_size)

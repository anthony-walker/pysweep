
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
    import pycuda.gpuarray as gpuarray
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
sys.path.insert(0, '/home/walkanth/pysweep/src/equations')
import euler

#Debuggin imports
import warnings
from euler import *
from plot_sweep_tests import *
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS



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
#------------------Shared Memory Arrays----------------------------#
cpu_array = None
gpu_array = None
#------------------Architecture-----------------------------------------
gpu_id = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10) #getting devices by load
dev_attrs = getDeviceAttrs(gpu_id[0],False)   #Getting device attributes
gpu = cuda.Device(gpu_id[0])    #Getting device with id via cuda
cores = mp.cpu_count()  #Getting number of cpu cores
root_cores = int(np.sqrt(cores))
#-----------------------------Functions----------------------------#
cpu_test_fcn = None
#-------------------------------Constants--------------------------#
SS = 0  #number of steps in Octahedron (Half Pyramid and DownPyramid)
#----------------------End Global Variables------------------------#

def sweep(arr0,targs,ops,block_size, cpu_fcn, gpu_fcn ,gpu_aff=None):
    """Use this function to perform swept rule
    args:
    arr0 -  2D numpy array of initial conditions
    targs - (t0,tf,dt) dt
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
    plane_shape = np.shape(arr0)  #Shape of initial conditions
    max_swept_step = min(block_size[2:])/(2*ops)
    time_steps = int((targs[1]-targs[0])/targs[2])
    arr = np.zeros((time_steps,)+arr0.shape,dtype=np.float32) #4D arr(t,x,y,v)
    arr[0] = arr0
    #----------------Reading In Source Code-------------------------------#
    source_code = source_code_read("./src/equations/euler.h")
    source_code += "\n"+source_code_read("./src/sweep/sweep.h")
    mod = SourceModule(source_code)
    # print(source_code)

    #------------------Dividing work based on the architecture----------------#
    if rank == master_rank:
        arr0s = rank_split(arr0,plane_shape,num_ranks)
        if True:
            gpu_speed(arr, mod, cpu_fcn, block_size,ops)
            # pass
        if False:
            cpu_speed(arr, cpu_fcn, block_size,ops)
    #----------------------------CUDA ------------------------------#

    #-------------------------END CUDA ---------------------------#

# (gpu_sum,cpu_sum,gpu_id) = arch_query()    #This gets architecture information from ranks
# gpu_affinity *= gpu_sum/cpu_sum #Adjust affinity by number of gpu/cpu_cores
# print(gpu_affinity)
# gpu_blocks,cpu_blocks = arch_work_blocks(plane_shape,block_size,gpu_affinity)
# print(gpu_blocks,cpu_blocks)
# print("Max: ",max_swept_step)

def gpu_speed(arr,source_mod,cpu_fcn,block_size,ops):
    """Use this function to measure the gpu's performance."""
    #Constants
    int_cast = lambda x:np.int32(x)
    const_dict = {'nv':(int_cast,arr.shape[1])}
    constant_copy(source_mod,**const_dict)

    grid_size = (int(arr.shape[2]/block_size[0]),int(arr.shape[3]/block_size[1]))   #Grid size
    shared_size = arr[0,:,:block_size[0],:block_size[1]].nbytes #Creating size of shared array
    arr = arr.astype(np.float32)
    # print(arr.shape)
    # print(grid_size)
    # print(block_size)
    # print(shared_size)
    gpu_fcn = source_mod.get_function("UpPyramid")
    # print(arr[0])
    # print("Split")
    gpu_fcn(cuda.InOut(arr),grid=grid_size, block=block_size,shared=shared_size)
    # print(arr[0])

def cpu_speed(arr,cpu_fcn,block_size,ops):
    """This function compares the speed of a block calculation to determine the affinity."""

    num_tries = 1 #Number of tries to get performance ratio
    sx = arr.shape[1]/block_size[0] #split x dimensions
    sy = arr.shape[2]/block_size[1] #split y dimension
    num_blocks = sx*sy  #number of blocks
    hcs = lambda x,s: np.array_split(x,s,axis=x.shape.index(max(x.shape)))   #Lambda function for creating blocks for cpu testing
    cpu_test_blocks =  [item for subarr in hcs(arr,sx) for item in hcs(subarr,sy)]    #applying lambda function
    cpu_test_fcn = sweep_lambda((UpPyramid,cpu_fcn,0,ops)) #Creating a function that can be called with only the block list
    #------------------------Testing CPU Performance---------------------------#
    pool = mp.Pool(cores)   #Pool allocation
    cpu_performance = 0
    for i in range(num_tries):
        start_cpu = time.time()
        cpu_res = pool.map_async(cpu_test_fcn,cpu_test_blocks)
        nts = cpu_res.get()
        stop_cpu = time.time()
        cpu_performance += np.prod(cpu_test_blocks[0].shape)*num_blocks/(stop_cpu-start_cpu)
    pool.close()
    pool.join()
    cpu_performance /= num_tries
    print("Average CPU Performance:", cpu_performance)
    return cpu_performance

    #-------------------------Ending CPU Performance Testing--------------------#
    #
    # #------------------------Testing GPU Performance---------------------------#
    # #Array
    # gpu_test_blocks =  np.ones(block_size,dtype=np.float64)*2
    # # print(gpu_test_blocks)
    # print(type(block_size))
    # start_gpu = cuda.Event()
    # stop_gpu = cuda.Event()
    # start_gpu.record()
    # gpu_dummy_fcn = source_mod.get_function("dummy_fcn")
    # gpu_dummy_fcn(cuda.InOut(gpu_test_blocks),grid=grid_size,block=block_size)
    # stop_gpu.record()
    # stop_gpu.synchronize()
    # gpu_performance = np.prod(grid_size)*np.prod(block_size)/(start_gpu.time_till(stop_gpu)*1e-3 )
    # # print(gpu_test_blocks)
    # performance_ratio = gpu_performance/cpu_performance
    # print(performance_ratio)
    #-------------------------Ending CPU Performance Testing--------------------#


#Swept time space decomposition CPU functions
def UpPyramid(arr, fcn, ts, ops):
    """This is the starting pyramid."""
    plane_shape = np.shape(arr)
    iidx = list(np.ndindex(plane_shape[1:-1]))
    #Bounds
    lb = 0
    ub = [plane_shape[1],plane_shape[2]]
    #Going through all swept steps
    t_start = ts
    # pts = [iidx]    #This is strictly for debugging
    while ub[0] > lb and ub[1] > lb:
        lb += ops
        ub = [x-ops for x in ub]
        iidx = [x for x in iidx if x[0]>=lb and x[1]>=lb and x[0]<ub[0] and x[1]<ub[1]]
        if iidx:
            step(arr,iidx,0)
            # pts.append(iidx) #This is strictly for debuggings
        ts+=1
    return ts
    # return pts #This is strictly for debugging

def Octahedron(arr, cpu_fcn, ts, ops):
    """This is the steps in between UpPyramid and DownPyramid."""
    pass

def DownPyramid(arr, cpu_fcn, ts, ops):
    """This is the ending inverted pyramid."""
    pass

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

def create_blocks(arr,block_size):
    """Use this function to create blocks from an array based on the given blocksize."""
    sx = arr.shape[1]/block_size[0] #split x dimensions
    sy = arr.shape[2]/block_size[1] #split y dimension
    num_blocks = sx*sy  #number of blocks
    hcs = lambda x,s: np.array_split(x,s,axis=x.shape.index(max(x.shape[1:])))   #Lambda function for creating blocks for cpu testing
    blocks =  [item for subarr in hcs(arr,sx) for item in hcs(subarr,sy)]    #applying lambda function
    return blocks

def constant_copy(source_mod,**const_kwargs):
    """Use this function to copy constant args to cuda memory."""
    for key in const_kwargs:
        c_ptr,_ = source_mod.get_global(key)
        cuda.memcpy_htod(c_ptr,const_kwargs[key][0](const_kwargs[key][1]))

def rank_split(arr0,plane_shape,rank_size):
    """Use this function to equally split data along the largest axis for ranks."""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(arr0,rank_size,axis=major_axis)


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

class sweep_lambda(object):
    """This class is a function wrapper to create kind of a pickl-able lambda function."""
    def __init__(self,args):
        self.args = args
    def __call__(self,x):
        sweep_fcn,cpu_fcn,ts,ops = self.args
        return sweep_fcn(x,cpu_fcn,ts,ops)

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



if __name__ == "__main__":
    # print("Starting execution.")
    dims = (4,int(128),int(128))
    arr0 = np.zeros(dims)
    arr0[1,:,:] = 1.0
    arr0[2,:,:] = 2.0
    arr0[3,:,:] = 3.0

    block_size = (32,32,1)
    dy = [0.1,0.1]
    t0 = 0
    t_b = 1
    dt = 0.1
    targs = (t0,t_b,dt)
    order = 2

    sweep(arr0,targs,order,block_size,euler.step,0)

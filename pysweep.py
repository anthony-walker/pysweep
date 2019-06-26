
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import os
import sys

#DataStructure and Math imports
import math
import numpy as np
from collections import deque

from pycuda.compiler import SourceModule
#Parallel programming imports
try:
   import pycuda.driver as cuda
   import pycuda.autoinit
   from pycuda.compiler import SourceModule

except:
   pass

import GPUtil
from mpi4py import MPI
#multiprocessing
import multiprocessing as mp
import ctypes

#Testing imports
import platform
import time

#--------------------Global Variables------------------------------#
shared_array = None

def sweep(y0, dy, t0, t_b, dt,ops,block_size,gpu_aff=None):
    """Use this function to perform swept rule>"""
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
    try:
        source_code = source_code_read("./csrc/swept_source.cu")
        source_code += "\n"+source_code_read("./csrc/euler_source.cu")
    except:
        source_code = source_code_read("./pysweep/csrc/swept_source.cu")
        source_code += "\n"+source_code_read("./pysweep/csrc/euler_source.cu")
    mod = SourceModule(source_code)

    #------------------Architecture-----------------------------------------
    gpu_id = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
    gpu = cuda.Device(gpu_id[0])
    cores = mp.cpu_count()

    #------------------Dividing work based on the architecture----------------#
    if rank == master_rank:
        y0s = rank_split(y0,plane_shape,num_ranks)
        if gpu_aff is None:
            arch_speed_comp(mod,dummy_fcn,block_size)

        # (gpu_sum,cpu_sum,gpu_id) = arch_query()    #This gets architecture information from ranks
        # gpu_affinity *= gpu_sum/cpu_sum #Adjust affinity by number of gpu/cpu_cores
        # print(gpu_affinity)
        # gpu_blocks,cpu_blocks = arch_work_blocks(plane_shape,block_size,gpu_affinity)
        # print(gpu_blocks,cpu_blocks)
        # print("Max: ",max_swept_step)

    # #----------------------------CUDA TESTING------------------------------#
    # #Array
    # time_len = 2
    #
    # a =  np.ones((2,2,time_len,2),dtype=np.float32)
    # a[:,:,0,:] = [2,3]
    #
    # #Getting cuda source

    #Creating cuda source

    #
    # #Constants
    # dt = np.array([0.01,],dtype=np.float32)
    # mss = np.array([100,],dtype=int)
    #
    # dt_ptr,_ = mod.get_global("dt")
    # mss_ptr,_ = mod.get_global("mss")
    #
    # #Copying to GPU memory
    # cuda.memcpy_htod(mss_ptr,mss)
    # cuda.memcpy_htod(dt_ptr,dt)
    #
    # #Executing cuda source
    # func = mod.get_function("sweep")
    # func(cuda.InOut(a),grid=(1,1,1),block=(2,2,1))
    # for x in a:
    #     for y in x:
    #         print("new time level")
    #         for t in y:
    #             print(t)
    #-------------------------END CUDA TESTING---------------------------#


def arch_speed_comp(source_mod,cpu_fcn,block_size):
    """This function compares the speed of a block calculation to determine the affinity."""
    time_steps = (2,)
    block_size = (5,2,2)+time_steps
     #----------------------------Testing OMP-------------------------------#
    shm_dim = int(np.prod(block_size))
    cores = mp.cpu_count()-14
    pool = mp.Pool(cores)
    # #Original random array
    rand_arr_o = np.random.random_sample(block_size)    #Creating random array
    # #Reshaping random array for shared memory
    # rand_arr_1 = np.reshape(rand_arr_o.copy(),shm_dim)
    # #Creating shared array
    # shared_arr = mp.Array(c.c_double,rand_arr_1)
    #Creating shared memory array
    global shared_array
    shared_array_base = mp.Array(ctypes.c_double, shm_dim)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(block_size)
    print("Woo")
    #Calling pool from number of cores with the dummy function
    # res = pool.map(cpu_fcn,range(10))

    # rand_arr_2 = np.frombuffer(shared_arr.get_obj())
    #
    # rand_arr_2 = np.reshape(rand_arr_2,block_size)
    #
    #
    # start_omp = time.time()
    # inds_itble = np.ndindex(block_size[:-1])


def dummy_fcn(val):
    """This is a testing function for arch_speed_comp."""
    print(val*val)
    return val*val

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


if __name__ == "__main__":
    # print("Starting execution.")
    dims = (int(20*256),int(5*256),4)
    y0 = np.zeros(dims)+1
    dy = [0.1,0.1]
    t0 = 0
    t_b = 1
    dt = 0.001
    order = 2
    block_size = (256,256,1)
    GA = 40
    sweep(y0,dy,t0,t_b,dt,2,block_size)

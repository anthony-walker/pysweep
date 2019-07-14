
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import os
import sys

#DataStructure and Math imports
import math
import numpy as np
from collections import deque


#CUDA Imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#MPI imports
from mpi4py import MPI

#Multiprocessing Imports
import multiprocessing as mp
import ctypes

#GPU Utility Imports
import GPUtil

#Swept imports
from pysweep_lambda import sweep_lambda
from pysweep_dev import *
from pysweep_functions import *

#Testing and Debugging
import platform
import time
sys.path.insert(0, '/home/walkanth/pysweep/src/equations')
import euler
import warnings
from euler import *
from pysweep_plot_tests import *
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS
#--------------------Global Variables------------------------------#

#----------------------End Global Variables------------------------#

def sweep(arr0,targs,dx,dy,ops,block_size,kernel_source,affinity=1,dType = np.dtype('float32')):
    """Use this function to perform swept rule
    args:
    arr0 -  3D numpy array of initial conditions (v (variables), x,y)
    targs - (t0,tf,dt)
    dx - spatial step on x axis
    dy - spatial step on y axis
    ops - number of atomic operations
    block_size - gpu block size (check your architecture requirements)
    affinity -  the GPU affinity (GPU work/CPU work)/TotalWork
    """

    #Getting GPU info
    gpu_ids = GPUtil.getAvailable(order = 'load',excludeID=[],limit=10000) #getting devices by load

    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    gpu_rank = False    #Says whether or not the rank is a GPU rank
    deviceID = None
    rank_info = comm.allgather((rank,processor,gpu_ids,gpu_rank,deviceID))

    #Max Swept Step
    MPSS = int(min(block_size[:-1])/(2*ops)) #Max Pyramid Swept Step
    MOSS = int(min(block_size[:-1])/(ops))   #Max Octahedron Swept Step
    SPLIT = int(block_size[0]/2)    #Split computation shift

    #----------------Data Input setup -------------------------#
    time_steps = int((targs[1]-targs[0])/targs[2])+1  #Number of time steps +1 becuase initial time step
    itemsize = int(dType.itemsize)

    #Determine which ranks are GPU ranks and create shared array data
    if rank == master_rank:
        new_rank_info, total_gpus = get_gpu_ranks(rank_info)
        total_gpus = num_ranks if num_ranks<total_gpus else total_gpus
        total_cpus = num_ranks-total_gpus
        # total_cpus = total_cpus if total_cpus%2==0 else total_cpus-total_cpus%2 #Making total cpus divisible by 2
        gpu_ranks = list()
        cpu_ranks = list()
        for ri in new_rank_info:
            if ri[3]:
                gpu_ranks.append(ri[0])
            else:
                cpu_ranks.append(ri[0])
        #Getting slices for data
        gpu_slices,cpu_slices = affinity_split(affinity,block_size,arr0.shape,total_gpus)
        gpu_arr0 = arr0[gpu_slices] #GPU initial array
        cpu_arr0 = arr0[cpu_slices] #CPU initial array
        #Creating new shapes
        new_cpu_shape = (MOSS,)+extended_shape(cpu_arr0.shape,block_size) if len(cpu_arr0[0])>0 else 0
        new_gpu_shape = (MOSS,)+extended_shape(gpu_arr0.shape,block_size) if len(gpu_arr0[0])>0 else 0
        gpu_arr_bytes = int(np.prod(new_gpu_shape)*itemsize) #4D arr(t,v,x,y)
        cpu_arr_bytes = int(np.prod(new_cpu_shape)*itemsize) #4D arr(t,v,x,y)
    else:
        new_rank_info = None
        total_gpus=None
        total_cpus = None
        cpu_arr_bytes = 0
        gpu_arr_bytes = 0
        new_cpu_shape=None
        new_gpu_shape=None
        gpu_ranks = None
        cpu_ranks = None

    #Scattering new rank information
    new_rank_info = comm.scatter(new_rank_info)
    #Updating gpu_rank bools
    gpu_rank = new_rank_info[3]
    #Broadcasting new shapes and number of gpus
    new_gpu_shape = comm.bcast(new_gpu_shape,root=master_rank)
    new_cpu_shape = comm.bcast(new_cpu_shape,root=master_rank)
    total_gpus = comm.bcast(total_gpus,root=master_rank)
    total_cpus = comm.bcast(total_cpus,root=master_rank)
    gpu_ranks = comm.bcast(gpu_ranks,root=master_rank)
    cpu_ranks = comm.bcast(cpu_ranks,root=master_rank)

    #New master ranks for each architecture
    gpu_master_rank = gpu_ranks[0] if len(gpu_ranks)>0 else None
    cpu_master_rank = cpu_ranks[0] if len(cpu_ranks)>0 else None

    #Creating GPU group comm
    gpu_group = comm.group.Incl(gpu_ranks)
    gpu_comm = comm.Create_group(gpu_group)

    #Creating CPU group comm
    cpu_group = comm.group.Incl(cpu_ranks)
    cpu_comm = comm.Create_group(cpu_group)

    #Grouping architecture arguments
    gargs = (gpu_comm,gpu_master_rank,new_gpu_shape,total_gpus,gpu_rank)
    cargs = (cpu_comm,cpu_master_rank,new_cpu_shape,total_cpus)

    #Create shared CPU array
    if np.prod(new_cpu_shape)>0:
        shared_cpu_arr = create_CPU_sarray(comm,new_cpu_shape,dType,cpu_arr_bytes)
        #Setting initial conditions
        if rank == master_rank:
            shared_cpu_arr[0,:,SPLIT:,:] = cpu_arr0[:,:,:]
        comm.Barrier()  #Ensure that all initial data has been written to the shared array and sync ranks

    #Getting region for each rank (CPU or GPU)
    region = region_split(rank,gargs,cargs,block_size)

    #GPU Swept Calculations
    # if gpu_rank:
    #     #----------------Reading In Source Code-------------------------------#
    #     source_code = source_code_read("/home/walkanth/pysweep/src/sweep/sweep.h")
    #     split_source_code = source_code.split("//!!(@#\n")
    #     source_code = split_source_code[0]+"\n"+source_code_read(kernel_source)+"\n"+split_source_code[1]
    #     source_mod = SourceModule(source_code)#,options=["--ptxas-options=-v"])
    # else:   #CPU Swept Calculations
    #     pass


    # #----------------------------Creating shared arrays-------------------------#
    # global cpu_array
    # cpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    # cpu_array = np.ctypeslib.as_array(cpu_array_base.get_obj())
    # cpu_array = cpu_array.reshape(block_size)
    #
    # global gpu_array
    # gpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    # gpu_array = np.ctypeslib.as_array(gpu_array_base.get_obj())
    # gpu_array = gpu_array.reshape(block_size)


def rank_split(arr0,rank_size):
    """Use this function to equally split data among the ranks"""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(arr0,rank_size,axis=major_axis)

if __name__ == "__main__":
    # print("Starting execution.")
    dims = (4,int(256),int(256))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 0.1
    arr0[1,:,:] = 0.0
    arr0[2,:,:] = 0.2
    arr0[3,:,:] = 0.125

    #GPU Arguments
    block_size = (32,32,1)
    kernel = "/home/walkanth/pysweep/src/equations/euler.h"
    affinity = 0.8
    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.01
    targs = (t0,t_b,dt)

    #Space testing arguments
    dx = 0.1
    dy = 0.1
    ops = 2

    sweep(arr0,targs,dx,dy,ops,block_size,kernel,affinity)

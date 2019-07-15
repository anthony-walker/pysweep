
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
# import pycuda.autoinit
from pycuda.compiler import SourceModule
# import pycuda.gpuarray as gpuarray

#MPI imports
from mpi4py import MPI

#Multiprocessing Imports
import multiprocessing as mp
import ctypes

#GPU Utility Imports
import GPUtil

#Swept imports
from .pysweep_lambda import sweep_lambda
from .pysweep_dev import *
from .pysweep_functions import *
import importlib.util
#Testing and Debugging
# import platform
# import time
# sys.path.insert(0, '/home/walkanth/pysweep/src/equations')
# import euler
# import warnings
# from euler import *
# from pysweep_plot_tests import *
# warnings.simplefilter("ignore") #THIS IGNORES WARNINGS
#--------------------Global Variables------------------------------#

#----------------------End Global Variables------------------------#

def sweep(arr0,targs,dx,dy,ops,block_size,gpu_source,cpu_source,affinity=1,dType = np.dtype('float32'),add_consts=dict()):
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

    #Global swept step
    GST = 0 #This is a variable to track which swept step the simulation is on

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

    #Max Swept Steps from block_size and other constants
    MPSS = int(min(block_size[:-1])/(2*ops)) #Max Pyramid Swept Step
    MOSS = int(min(block_size[:-1])/(ops))   #Max Octahedron Swept Step
    SPLITX = int(block_size[0]/2)    #Split computation shift
    SPLITY = int(block_size[1]/2)    #Split computation shift
    #----------------Data Input setup -------------------------#
    time_steps = int((targs[1]-targs[0])/targs[2])+1  #Number of time steps +1 becuase initial time step
    itemsize = int(dType.itemsize)
    #Create shared process array for data transfer
    shared_shape = (MOSS,arr0.shape[0],arr0.shape[1]+SPLITX,arr0.shape[2]+SPLITY)
    shared_arr = create_CPU_sarray(comm,shared_shape,dType,np.prod(shared_shape)*itemsize)
    #Setting initial conditions
    if rank == master_rank:
        shared_arr[0,:,:arr0.shape[1],:arr0.shape[2]] = arr0[:,:,:]
    comm.Barrier()  #Ensure that all initial data has been written to the shared array and sync ranks


    #Determine which ranks are GPU ranks and create shared array data
    if rank == master_rank:
        new_rank_info, total_gpus = get_gpu_ranks(rank_info)
        total_gpus = num_ranks if num_ranks<total_gpus else total_gpus
        total_cpus = num_ranks-total_gpus
        affinity = affinity if total_cpus != 0 else 1   #sets affinity to 1 if there are no cpus avaliable
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
        gpu_shape = get_slices_shape(gpu_slices)
        cpu_shape =  get_slices_shape(cpu_slices)

    else:
        new_rank_info = None
        total_gpus=None
        total_cpus = None
        gpu_ranks = None
        cpu_ranks = None
        gpu_shape = None
        cpu_shape = None

    #Scattering new rank information
    new_rank_info = comm.scatter(new_rank_info)
    #Updating gpu_rank bools
    gpu_rank = new_rank_info[3]
    #Broadcasting new shapes and number of gpus
    total_gpus = comm.bcast(total_gpus,root=master_rank)
    total_cpus = comm.bcast(total_cpus,root=master_rank)
    gpu_ranks = comm.bcast(gpu_ranks,root=master_rank)
    cpu_ranks = comm.bcast(cpu_ranks,root=master_rank)
    gpu_shape = comm.bcast(gpu_shape,root=master_rank)
    cpu_shape = comm.bcast(cpu_shape,root=master_rank)

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
    gargs = (gpu_comm,gpu_master_rank,gpu_shape,total_gpus,gpu_rank)
    cargs = (cpu_comm,cpu_master_rank,cpu_shape,total_cpus)

    #Getting region and offset region for each rank (CPU or GPU)
    regions = region_split(rank,gargs,cargs,block_size,MOSS)

    #Initializing local array and filling with arr0
    # print(rank,regions[0])
    local_shape = get_slices_shape(regions[0])
    local_array = np.zeros(local_shape,dtype=dType)
    local_array[:,:,:,:] = shared_arr[regions[0]]

    #Specifying which source mod to use
    if gpu_rank:
        #Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(new_rank_info[-1])
        cuda_context = cuda_device.make_context()
        #Creating grid size
        grid_size = (int(local_shape[2]/block_size[0]),int(local_shape[3]/block_size[1]))   #Grid size
        #Creating constants
        NV = local_shape[1]
        SGIDS = block_size[0]*block_size[1]
        VARS =  SGIDS*grid_size[0]*grid_size[1]
        TIMES = VARS*NV
        DX = dx
        DY = dy
        DT = targs[2]
        const_dict = ({"NV":NV,"DX":DX,"DT":DX,"DY":DY,"SGIDS":SGIDS,"VARS":VARS
                    ,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS})
        #Building CUDA source code
        source_mod = build_gpu_source(gpu_source)
        # constant_copy(source_mod,const_dict,add_consts)
    else:
        source_mod = build_cpu_source(cpu_source) #Building Python source code


    # print(rank,local_array.shape)
    #First step is the up pyramid
    # UpPyramid(source_mod,local_array, ops, gpu_rank,block_size)
    # GST+=1  #Increment global swept step
    # comm.Barrier()

    #CUDA clean up - One of the last steps
    if gpu_rank:
        cuda_context.pop()

def rank_split(arr0,rank_size):
    """Use this function to equally split data among the ranks"""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(arr0,rank_size,axis=major_axis)

if __name__ == "__main__":
    # print("Starting execution.")
    dims = (4,int(32),int(32))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 0.1
    arr0[1,:,:] = 0.5
    arr0[2,:,:] = 0.2
    arr0[3,:,:] = 0.125

    #GPU Arguments
    block_size = (8,8,1)
    kernel = "/home/walkanth/pysweep/src/equations/euler.h"
    cpu_source = "/home/walkanth/pysweep/src/equations/euler.py"
    affinity = 0.9
    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.01
    targs = (t0,t_b,dt)

    #Space testing arguments
    dx = 0.1
    dy = 0.1
    ops = 2

    sweep(arr0,targs,dx,dy,ops,block_size,kernel,cpu_source,affinity)

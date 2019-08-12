#Programmer: Anthony Walker
#This file contains the implementation of a standard PDE solve in parallel
#This is for comparison to the swept rule code


#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import os
import sys

#Writing imports
import h5py

#DataStructure and Math imports
import math
import numpy as np
from collections import deque

#CUDA Imports
import pycuda.driver as cuda
import pycuda.autoinit  #Cor debugging only
from pycuda.compiler import SourceModule
# import pycuda.gpuarray as gpuarray

#MPI imports
from mpi4py import MPI

#Multiprocessing Imports
import multiprocessing as mp
import ctypes

#GPU Utility Imports
import GPUtil
#Decomp functions
from .decomp_functions import *

def decomp(arr0,targs,dx,dy,ops,block_size,gpu_source,cpu_source,affinity=1,dType=np.dtype('float32'),filename = "results",add_consts=dict()):
    """Use this function to perform standard decomposition
    args:
    arr0 -  3D numpy array of initial conditions (v (variables), x,y)
    targs - (t0,tf,dt)
    dx - spatial step on x axis
    dy - spatial step on y axis
    ops - number of atomic operations
    block_size - gpu block size (check your architecture requirements)
    affinity -  the GPU affinity (GPU work/CPU work)/TotalWork
    """
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2

    #Getting GPU info
    gpu_ids = GPUtil.getAvailable(order = 'load',excludeID=[],limit=10000) #getting devices by load
        #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    master_rank = ZERO #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    gpu_rank = False    #Says whether or not the rank is a GPU rank
    #None is in place for device ID
    rank_info = comm.allgather((rank,processor,gpu_ids,gpu_rank,None))
    printer = pysweep_printer(rank,master_rank) #This is a printer to print only master rank data

    #----------------Data Input setup -------------------------#
    time_steps = int((targs[ONE]-targs[ZERO])/targs[TWO])+ONE  #Number of time steps +ONE becuase initial time step
    shared_shape = (time_steps,arr0.shape[ZERO],arr0.shape[ONE]+TWO*ops ,arr0.shape[TWO]+TWO*ops )
    shared_arr = create_CPU_sarray(comm,shared_shape,dType,np.prod(shared_shape)*itemsize)

    if rank == master_rank:
        new_rank_info, total_gpus = get_gpu_ranks(rank_info,affinity)
        #Adjusting if affinity is 0
        if affinity==ZERO:
            total_gpus = ZERO
        else:
            total_gpus = num_ranks if num_ranks<total_gpus else total_gpus
        #Getting total cpus
        total_cpus = num_ranks-total_gpus
        #Adjusting affinity if no cpus avaliable
        affinity = affinity if total_cpus != ZERO else ONE   #sets affinity to ONE if there are no cpus avaliable
        #Adjust for number of cpus/gpus
        # affinity*=(total_gpus/total_cpus)
        # affinity = affinity if affinity <= ONE else ONE #If the number becomes greater than ONE due to adjusting for number of cpus/gpus set to ONE
        # printer(affinity)
        # total_cpus = total_cpus if total_cpus%TWO==ZERO else total_cpus-total_cpus%TWO #Making total cpus divisible by TWO
        gpu_ranks = list()
        cpu_ranks = list()
        for ri in new_rank_info:
            if ri[3]:
                gpu_ranks.append(ri[ZERO])
            else:
                cpu_ranks.append(ri[ZERO])
        #Getting slices for data
        gpu_slices,cpu_slices = affinity_split(affinity,block_size,arr0.shape,total_gpus,printer)
    else:
        new_rank_info = None
        total_gpus=None
        total_cpus = None
        gpu_ranks = None
        cpu_ranks = None
        gpu_slices = None
        cpu_slices = None

    #Scattering new rank information
    new_rank_info = comm.scatter(new_rank_info)
    #Updating gpu_rank bools
    gpu_rank = new_rank_info[3]
    #Broadcasting new shapes and number of gpus
    total_gpus = comm.bcast(total_gpus,root=master_rank)
    total_cpus = comm.bcast(total_cpus,root=master_rank)
    gpu_ranks = comm.bcast(gpu_ranks,root=master_rank)
    cpu_ranks = comm.bcast(cpu_ranks,root=master_rank)
    gpu_slices = comm.bcast(gpu_slices,root=master_rank)
    cpu_slices = comm.bcast(cpu_slices,root=master_rank)

    #New master ranks for each architecture
    gpu_master_rank = gpu_ranks[ZERO] if len(gpu_ranks)>ZERO else None
    cpu_master_rank = cpu_ranks[ZERO] if len(cpu_ranks)>ZERO else None

    #Creating GPU group comm
    gpu_group = comm.group.Incl(gpu_ranks)
    gpu_comm = comm.Create_group(gpu_group)

    #Creating CPU group comm
    cpu_group = comm.group.Incl(cpu_ranks)
    cpu_comm = comm.Create_group(cpu_group)

    #Grouping architecture arguments
    gargs = (gpu_comm,gpu_master_rank,total_gpus,gpu_slices,gpu_rank)
    cargs = (cpu_comm,cpu_master_rank,total_cpus,cpu_slices)

    # Specifying which source mod to use
    if gpu_rank:
        # Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(new_rank_info[-ONE])
        cuda_context = cuda_device.make_context()
        #Creating local GPU array with split
        grid_size = (int(local_array.shape[TWO]/block_size[ZERO]),int(local_array.shape[3]/block_size[ONE]))   #Grid size
        #Creating constants
        NV = local_array.shape[ONE]
        # ARRX = local_array.shape[TWO]
        # ARRY = local_array.shape[3]
        SGIDS = (block_size[ZERO]+TWO*ops)*(block_size[ONE]+TWO*ops)
        VARS =  local_array.shape[TWO]*local_array.shape[3]
        TIMES = VARS*NV
        DX = dx
        DY = dy
        DT = targs[TWO]
        const_dict = ({"NV":NV,"DX":DX,"DT":DT,"DY":DY,"SGIDS":SGIDS,"VARS":VARS
                    ,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS,"OPS":ops})
        #Building CUDA source code
        source_mod = build_gpu_source(gpu_source)
        constant_copy(source_mod,const_dict,add_consts)
        local_cpu_regions = None
    else:
        source_mod = build_cpu_source(cpu_source) #Building Python source code
        local_cpu_regions = create_blocks_list(local_array,block_size,ops,printer)
        grid_size = None

    #Setting local array boolean
    if local_array is None:
        LAB = False
    else:
        LAB = True

    #Creating HDF5 file
    if LAB:
        filename+=".hdf5"
        hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        hdf5_data_set = hdf5_file.create_dataset("data",(nts,local_array.shape[ONE],arr0.shape[ONE],arr0.shape[TWO]),dtype=dType)

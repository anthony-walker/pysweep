
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

#Swept imports
from .pysweep_lambda import sweep_lambda
from .pysweep_dev import *
from .pysweep_functions import *
from .pysweep_printer import pysweep_printer
import importlib.util
#Testing and Debugging
# import platform
# import time
# sys.path.insert(0, '/home/walkanth/pysweep/src/equations')
# import euler
import warnings
# from euler import *
# from pysweep_plot_tests import *
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS
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
    #Max Swept Steps from block_size and other constants
    MPSS = int(min(block_size[:-ONE])/(TWO*ops+ONE))+ONE #Max Pyramid Swept Step
    MOSS = int(TWO*min(block_size[:-ONE])/(TWO*ops+ONE))   #Max Octahedron Swept Step
    # printer(MPSS)
    #Splits for shared array
    SPLITX = int(block_size[ZERO]/TWO)   #Split computation shift - add ops
    SPLITY = int(block_size[ONE]/TWO)   #Split computation shift

    #----------------Data Input setup -------------------------#
    time_steps = int((targs[ONE]-targs[ZERO])/targs[TWO])+ONE  #Number of time steps +ONE becuase initial time step
    itemsize = int(dType.itemsize)
    #Global swept step
    GST = ZERO #This is a variable to track which swept step the simulation is on
    MGST =   TWO if time_steps<MOSS else int(np.ceil(time_steps/MOSS))

    #Create shared process array for data transfer
    shared_shape = (MOSS,arr0.shape[ZERO],arr0.shape[ONE]+SPLITX+TWO*ops ,arr0.shape[TWO]+SPLITY+TWO*ops )
    shared_arr = create_CPU_sarray(comm,shared_shape,dType,np.prod(shared_shape)*itemsize)
    """PRINTER"""
    # printer(shared_shape)
    #Setting initial conditions
    if rank == master_rank:
        shared_arr[ZERO,:,ops:arr0.shape[ONE]+ops,ops:arr0.shape[TWO]+ops] = arr0[:,:,:]
        #Update edges
        edge_comm(shared_arr,SPLITX,SPLITY,ops,ZERO)
    comm.Barrier()  #Ensure that all initial data has been written to the shared array and sync ranks

    #Determine which ranks are GPU ranks and create shared array data
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
    #Getting region and offset region for each rank (CPU or GPU)
    regions = create_write_regions(rank,gargs,cargs,block_size,ops,MOSS,SPLITX,SPLITY,printer)
    regions = create_read_regions(regions,ops)+regions
    #Regions: (read region, shifted read, write, shifted write)
    local_array = local_array_create(shared_arr,regions[ZERO],dType)
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
        SGIDS = block_size[ZERO]*block_size[ONE]+TWO*ops
        VARS =  SGIDS*grid_size[ZERO]*grid_size[ONE]
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
    # printer(local_array[0,0,:,:])
    #Creating swept indexes
    idx_sets = create_iidx_sets(block_size,ops)
    # if rank == master_rank:
    #     for i in idx_sets[0]:
    #         shared_arr[0,0,i[0],i[1]] = 0
    #     for i in idx_sets[1]:
    #         shared_arr[0,0,i[0],i[1]] = 5
    #     printer(shared_arr[0,0,local_cpu_regions[0][2],local_cpu_regions[0][3]])
    # print(local_cpu_regions)
    #UpPyramid Step
    if LAB:
        UpPyramid(source_mod,local_array,gpu_rank,block_size,grid_size,regions[TWO],local_cpu_regions,shared_arr,idx_sets,ops,printer) #THis modifies shared array
    comm.Barrier()
    # printer(shared_arr[ONE,ONE,:,:])
    # #Communicate edges
    # if LAB and rank == master_rank:
    #     edge_comm(shared_arr,SPLITX,SPLITY,GST%TWO)
    # comm.Barrier()
    # GST+=ONE  #Increment global swept step
    # Octahedron steps
    # while(GST<MGST):
    #     if LAB:
    #         cregion = regions[GST%TWO]
    #         #Reading local array
    #         local_array = shared_arr[cregion]
    #         #Octahedron Step
    #         Octahedron(source_mod,local_array, gpu_rank,block_size,grid_size,cregion,shared_arr,idx_sets)
    #     comm.Barrier()  #Write barrier
    #     if rank == master_rank:
    #         edge_comm(shared_arr,SPLITX,SPLITY,GST%TWO)
    #     comm.Barrier() #Edge transfer barrier
    #     GST+=ONE
    #     #Add JSON WRITE HERE and SHIFT
    #
    # #Down Pyramid Step
    # if LAB:
    #     DownPyramid(source_mod,local_array,gpu_rank,block_size,grid_size,cregion,shared_arr,idx_sets)
    #
    #Add Final Write Step Here

    #CUDA clean up - One of the last steps
    if gpu_rank:
        cuda_context.pop()

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
    affinity = 0
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

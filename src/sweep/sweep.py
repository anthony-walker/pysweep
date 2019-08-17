
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
from .pysweep_decomposition import *
import importlib.util
#Testing and Debugging
import warnings
import time as timer
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS
#--------------------Global Variables------------------------------#

#----------------------End Global Variables------------------------#

def sweep(arr0,targs,dx,dy,ops,block_size,gpu_source,cpu_source,affinity=1,dType=np.dtype('float32'),filename = "results",add_consts=dict(),exid=[]):
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
    start = timer.time()
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2
    TOPS = TWO*ops
    #Getting GPU info
        #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    master_rank = ZERO #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank

    printer = pysweep_printer(rank,master_rank)

    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Max Swept Steps from block_size and other constants
    mbx = block_size[0]
    mby = block_size[1]
    MPSS = 1 #Max Pyramid Swept Step  #Start at 1 to account for ghost node calc
    while(mbx>=(TOPS+ONE) and mby>=(TOPS+ONE)):
        mbx -= TOPS
        mby -= TOPS
        MPSS += 1
    MOSS = int(TWO*(MPSS-1))+1   #Max Octahedron Swept Step
    #Splits for shared array
    SPLITX = int(block_size[ZERO]/TWO)   #Split computation shift - add ops
    SPLITY = int(block_size[ONE]/TWO)   #Split computation shift

    #----------------Data Input setup -------------------------#
    time_steps = int((targs[ONE]-targs[ZERO])/targs[TWO])+ONE  #Number of time steps +ONE becuase initial time step
    #Global swept step
    GST = ZERO #This is a variable to track which swept step the simulation is on
    MGST = int(np.ceil(time_steps/MOSS))    #THIS ASSUMES THAT time_steps > MOSS
    time_steps = MGST*MOSS+1 #Updating time steps
    #-----------------------SHARED ARRAY CREATION----------------------#
    #Create shared process array for data transfer
    shared_shape = (MOSS+ONE,arr0.shape[ZERO],arr0.shape[ONE]+TOPS+SPLITX,arr0.shape[TWO]+TOPS+SPLITY)
    shared_arr = create_CPU_sarray(comm,shared_shape,dType,np.prod(shared_shape)*dType.itemsize)
    printer(shared_shape)
    #-----------------------------INITIAL SPLIT OF DOMAIN------------------------------------#
    if rank == master_rank:
        #Determining the split between gpu and cpu
        gpu_slices,cpu_slices = get_affinity_slices(affinity,block_size,arr0.shape)
        #Getting gpus
        if affinity>0:
            gpu_rank = GPUtil.getAvailable(order = 'load',excludeID=exid,limit=1e8) #getting devices by load
            gpu_rank = [(True,id) for id in gpu_rank]
            num_gpus = len(gpu_rank)
        else:
            gpu_rank = []
            num_gpus = 0
        gpu_rank += [(False,None) for i in range(num_gpus,num_ranks)]
        gpu_ranks = np.array([i for i in range(ZERO,num_gpus)])
        cpu_ranks = np.array([i for i in range(num_gpus,num_ranks)])
        #Update boundary points

    else:
        gpu_rank = None
        gpu_ranks = None
        cpu_ranks = None
        gpu_slices = None
        cpu_slices = None

    #-------------------------------INITIAL COLLECTIVE COMMUNICATION
    gpu_rank = comm.scatter(gpu_rank,root=master_rank)
    gpu_ranks = comm.bcast(gpu_ranks,root=master_rank)
    cpu_ranks = comm.bcast(cpu_ranks,root=master_rank)
    gpu_slices = comm.bcast(gpu_slices,root=master_rank)
    cpu_slices = comm.bcast(cpu_slices,root=master_rank)

    #GPU rank- MPI Variables
    if affinity > 0:
        gpu_group  = comm.group.Incl(gpu_ranks)
        gpu_comm = comm.Create_group(gpu_group)
        gpu_master = ZERO

    #Destroys cpu processes if affinity is one
    if affinity == 1:
        comm = comm.Create_group(gpu_group)
        if rank not in gpu_ranks:
            MPI.Finalize()
            exit(0)
    else:
        #CPU rank- MPI Variables
        cpu_group  = comm.group.Incl(cpu_ranks)
        cpu_comm = comm.Create_group(cpu_group)
        cpu_master = ZERO
    #Number of each architecture
    num_gpus = len(gpu_ranks)
    num_cpus = len(cpu_ranks)

    #------------------------------DECOMPOSITION BY REGION CREATION--------------#
    #Splits regions up amongst architecture
    if gpu_rank[0]:
        gri = gpu_comm.Get_rank()
        cri = None
        wregion = create_write_region(gpu_comm,gri,gpu_master,num_gpus,block_size,arr0.shape,gpu_slices,shared_shape[0],ops)
    else:
        cri = cpu_comm.Get_rank()
        gri = None
        wregion = create_write_region(cpu_comm,cri,cpu_master,num_cpus,block_size,arr0.shape,cpu_slices,shared_shape[0],ops)



    #---------------------_REMOVING UNWANTED RANKS---------------------------#
    #Checking if rank is useful
    if wregion[3].start == wregion[3].stop or wregion[2].start == wregion[2].stop:
        include_ranks = None
    else:
        include_ranks = rank
    #Gathering results
    include_ranks = comm.gather(include_ranks,root=master_rank)
    if rank == master_rank:
        include_ranks = list(filter(lambda a: a is not None, include_ranks))
    #Distributing new results
    include_ranks = comm.bcast(include_ranks,root=master_rank)
    #Creating new comm group
    comm_group = comm.group.Incl(include_ranks)
    #Killing unwanted processes
    if rank not in include_ranks:
        MPI.Finalize()
        exit(0)
    #Updating comm
    comm = comm.Create_group(comm_group)
    #Sync all processes - Ensures boundaries are updated
    comm.Barrier()
    #--------------------------CREATING OTHER REGIONS--------------------------#

    rregion = create_read_region(wregion,ops)   #Create read region
    srregion,xbregion,ybregion = create_shift_regions(rregion,SPLITX,SPLITY,shared_shape,ops)  #Create shifted read region
    boundaries,cregions = create_boundary_regions(wregion,SPLITX,SPLITY,ops,shared_shape)

    #Creating sets for CPU Calculations
    idx_sets = create_iidx_sets(block_size,ops)
    bridge_sets, bridge_dims = create_bridge_sets(mbx,mby,block_size,ops,MPSS)

    #--------------------------------CREATING LOCAL ARRAYS-----------------------------------------#
    local_array = create_local_array(shared_arr,rregion,dType)
    x_array = create_local_array(shared_arr,xbregion,dType)
    y_array = create_local_array(shared_arr,ybregion,dType)

    #---------------Generating Source Modules----------------------------------#
    if gpu_rank[0]:
        # Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank[ONE])
        cuda_context = cuda_device.make_context()
        #Creating local GPU array with split
        grid_size = (int((local_array.shape[TWO]-TOPS)/block_size[ZERO]),int((local_array.shape[3]-TOPS)/block_size[ONE]))   #Grid size
        #Creating constants
        NV = local_array.shape[ONE]
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
        cpu_regions = None
    else:
        source_mod = build_cpu_source(cpu_source) #Building Python source code
        cpu_regions = create_blocks_list(local_array.shape,block_size,ops)
        grid_size = None

    #------------------------------HDF5 File------------------------------------------#
    filename+=".hdf5"
    hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=comm)
    hdf_bs = hdf5_file.create_dataset("block_size",(len(block_size),))
    hdf_bs[:] = block_size[:]
    hdf_as = hdf5_file.create_dataset("array_size",(len(arr0.shape)+ONE,))
    hdf_as[:] = (time_steps,)+arr0.shape
    hdf_aff = hdf5_file.create_dataset("affinity",(ONE,))
    hdf_aff[ZERO] = affinity
    hdf_time = hdf5_file.create_dataset("time",(1,))
    hdf5_data_set = hdf5_file.create_dataset("data",(time_steps,arr0.shape[ZERO],arr0.shape[ONE],arr0.shape[TWO]),dtype=dType)
    hregion = (wregion[1],slice(wregion[2].start-ops,wregion[2].stop-ops,1),slice(wregion[3].start-ops,wregion[3].stop-ops,1))
    hdf5_data_set[0,hregion[0],hregion[1],hregion[2]] = shared_arr[0,wregion[1],wregion[2],wregion[3]]
    comm.Barrier() #Ensure all processes are prepared to solve

    # for b in boundaries:
    #     print(rank,b[2:])
    #     print(rank,wregion[2:],wregion[2:])
    #-------------------------------SWEPT RULE---------------------------------------------#
    # UpPyramid - shifts data appropriately
    UpPyramid(source_mod,local_array,gpu_rank[0],block_size,grid_size,wregion,boundaries,cpu_regions,shared_arr,idx_sets,ops) #THis modifies shared array
    comm.Barrier()
    print(bridge_sets[1])
    print(local_array.shape)

    # while(GST<=MGST):
    #Copy from the appropriate array
    x_array[:,:,:,:] = shared_arr[xbregion]
    y_array[:,:,:,:] = shared_arr[ybregion]
    #Bridge Step
    # Bridge(comm,source_mod,x_array,y_array,gpu_rank[0],block_size,grid_size,xbregion,ybregion,boundaries,cpu_regions,shared_arr,bridge_sets,ops,printer=printer) #THis modifies shared array
    # comm.Barrier()
    # printer(shared_arr[2,0,:,:])
    # for boundary in cregions:
    #     print(rank,boundary)
    # boundary_update(shared_arr,ops,cregions,ZERO)
    #
    # comm.Barrier()
    #
    # printer(shared_arr[2,0,:,:])
    # cregion = regions[(GST+1)%TWO] #This is the correct read region
    #Reading local array

    # local_array[:,:,:,:] = shared_arr[rregion]  #Array of first step seems to be correct
    # # printer(local_array[time_id,var_id,:,:])
    # #Octahedron Step
    # # Octahedron(source_mod,local_array,gpu_rank,block_size,grid_size,regions[(GST+1)%TWO+TWO],local_cpu_regions,shared_arr,idx_sets,ops)
    #
    # comm.Barrier()  #Write barrier
        #
        # if rank == master_rank and LAB:
        #     edge_comm(shared_arr,SPLITX,SPLITY,ops,(GST+1)%TWO)
        # comm.Barrier() #Edge transfer barrier
        # #Write and shift step
        # if LAB:
        #     write_and_shift(shared_arr,regions[TWO],hdf5_data_set,ops,MPSS,GST)
        # comm.Barrier()
        # GST+=ONE
        #
    # #Down Pyramid Step and final write
    # if LAB:
    #     DownPyramid(source_mod,local_array,gpu_rank,block_size,grid_size,regions[(GST+1)%TWO+TWO],local_cpu_regions,shared_arr,idx_sets,ops)
    #     write_and_shift(shared_arr,regions[TWO],hdf5_data_set,ops,MPSS,GST)

    comm.Barrier()
    #CUDA clean up - One of the last steps
    if gpu_rank[0]:
        cuda_context.pop()
    # Close file
    hdf5_file.close()

    # comm.Barrier()
    # stop = timer.time()
    # exec_time = abs(stop-start)
    # exec_time = comm.gather(exec_time)
    # if rank == master_rank:
    #     avg_time = sum(exec_time)/num_ranks
    #     hdf_time[0] = avg_time
    # comm.Barrier()

    # #This if for testing only
    # if rank == master_rank:
    #     return avg_time

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

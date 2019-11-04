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
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  #Cor debugging only
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
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
#Testing
import time as timer

def decomp(arr0,gargs,swargs,dType=np.dtype('float32'),filename ="results",exid=[]):
    """Use this function to perform standard decomposition
    args:
    arr0 -  3D numpy array of initial conditions (v (variables), x,y)
    gargs:   (global args)
        The first three variables must be time arguments (t0,tf,dt).
        Subsequent variables can be anything passed to the source code in the
        "set_globals" function
    swargs: (Swept args)
        TSO - Time scheme order
        OPS - operating points to remove from each side for a function step
            (e.g. a 5 point stencil would result in OPS=2).
        BS - gpu block size as integer which represents x and y dimensions
        AF -  the GPU affinity (GPU work/CPU work)/TotalWork
        GS - GPU source code file
        CS - CPU source code file
    dType: data type for the run (e.g. np.dtype("float32")).
    filename: Name of the output file excluding hdf5
    exid: GPU ids to exclude from the calculation.
    """
    t0,tf,dt = gargs[:3]
    TSO,OPS,BS,AF,GS,CS = swargs
    BS = (BS,BS,1)
    start = timer.time()
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2
    TOPS = TWO*OPS
    #Getting GPU info
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    master_rank = ZERO #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank

    #-----------------------------INITIAL SPLIT OF DOMAIN------------------------------------#
    if rank == master_rank:
        #Determining the split between gpu and cpu
        gpu_slices,cpu_slices = get_affinity_slices(AF,BS,arr0.shape)
        #Getting gpus
        if AF>0:
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
    if AF > 0:
        gpu_group  = comm.group.Incl(gpu_ranks)
        gpu_comm = comm.Create_group(gpu_group)
        gpu_master = ZERO

    #Destroys cpu processes if AF is one
    if AF == 1:
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
    #----------------Data Input setup -------------------------#
    time_steps = int((tf-t0)/dt)+ONE  #Number of time steps +ONE becuase initial time step
    #-----------------------SHARED [ARRAY] CREATION----------------------#
    #Create shared process array for data transfer  - TWO is added to shared shaped for IC and First Step
    shared_shape = (3,arr0.shape[ZERO],arr0.shape[ONE]+TOPS,arr0.shape[TWO]+TOPS)
    shared_arr = create_CPU_sarray(comm,shared_shape,dType,np.prod(shared_shape)*dType.itemsize)
    ssb = np.zeros((2,arr0.shape[ZERO],BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
    #Fill shared array and communicate initial boundaries
    if rank == master_rank:
        shared_arr[0,:,OPS:shared_shape[2]-OPS,OPS:shared_shape[3]-OPS] = arr0[:,:,:] #Filling array
        shared_arr[1,:,OPS:shared_shape[2]-OPS,OPS:shared_shape[3]-OPS] = arr0[:,:,:] #Filling array

    #------------------------------DECOMPOSITION BY REGION CREATION--------------#
    #Splits regions up amongst architecture
    if gpu_rank[0]:
        gri = gpu_comm.Get_rank()
        cri = None
        regions = create_write_region(gpu_comm,gri,gpu_master,num_gpus,BS,arr0.shape,gpu_slices,shared_shape[0],OPS)
    else:
        cri = cpu_comm.Get_rank()
        gri = None
        regions = create_write_region(cpu_comm,cri,cpu_master,num_cpus,BS,arr0.shape,cpu_slices,shared_shape[0],OPS)
    regions = (create_read_region(regions,OPS),regions) #Creating read region
    brs = create_boundary_regions(regions[1],shared_shape,OPS)
    comm.Barrier()

    reg_edge_comm(shared_arr,OPS,brs,regions[1])
    comm.Barrier()
    #Communicate edges
    local_array = np.copy(shared_arr[regions[0]])
    #---------------------_REMOVING UNWANTED RANKS---------------------------#
    #Checking if rank is useful
    if regions[ONE][3].start == regions[ONE][3].stop or regions[ONE][2].start == regions[ONE][2].stop:
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
    #---------------Generating Source Modules----------------------------------#
    if gpu_rank[0]:
        # Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank[ONE])
        cuda_context = cuda_device.make_context()
        #Creating local GPU array with split
        grid_size = (int((local_array.shape[TWO]-TOPS)/BS[ZERO]),int((local_array.shape[3]-TOPS)/BS[ONE]))   #Grid size
        #Creating constants
        NV = local_array.shape[ONE]
        SGIDS = (BS[ZERO]+TWO*OPS)*(BS[ONE]+TWO*OPS)
        STS = SGIDS*NV #Shared time shift
        VARS =  local_array.shape[TWO]*local_array.shape[3]
        TIMES = VARS*NV
        const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"OPS":OPS,"TSO":TSO,"STS":STS})
        #Building CUDA source code
        source_mod = build_gpu_source(GS)
        decomp_constant_copy(source_mod,const_dict)
        cpu_source_mod = build_cpu_source(CS)   #Building cpu source for set_globals
        cpu_source_mod.set_globals(gpu_rank[0],source_mod,*gargs)
        del cpu_source_mod
        decomp_set = None
    else:
        source_mod = build_cpu_source(CS) #Building Python source code
        decomp_set = create_decomp_sets(shared_arr[regions[0]].shape,OPS)
        source_mod.set_globals(gpu_rank[0],source_mod,*gargs)
        grid_size = None
    #------------------------------HDF5 File------------------------------------------#
    filename+=".hdf5"
    hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=comm)
    hdf_bs = hdf5_file.create_dataset("BS",(len(BS),))
    hdf_bs[:] = BS[:]
    hdf_as = hdf5_file.create_dataset("array_size",(len(arr0.shape)+ONE,))
    hdf_as[:] = (time_steps,)+arr0.shape
    hdf_aff = hdf5_file.create_dataset("AF",(ONE,))
    hdf_aff[ZERO] = AF
    hdf_time = hdf5_file.create_dataset("time",(1,))
    hdf5_data_set = hdf5_file.create_dataset("data",(time_steps+1,arr0.shape[ZERO],arr0.shape[ONE],arr0.shape[TWO]),dtype=dType)
    hregion = (regions[ONE][1],slice(regions[ONE][2].start-OPS,regions[ONE][2].stop-OPS,1),slice(regions[ONE][3].start-OPS,regions[ONE][3].stop-OPS,1))
    hdf5_data_set[0,hregion[0],hregion[1],hregion[2]] = shared_arr[0,regions[ONE][1],regions[ONE][2],regions[ONE][3]]
    wr = regions[1]
    #Solution
    ct = 1
    comm.Barrier()
    #Solving loop
    for i in range(0, TSO*(time_steps)):
        local_array = np.copy(shared_arr[regions[0]])
        decomposition(source_mod,local_array, gpu_rank[0], BS, grid_size,regions[ONE],decomp_set,shared_arr,OPS,i,TSO,ssb)
        comm.Barrier()
        #Writing Data after it has been shifted
        if (i+1)%TSO==0:
            hdf5_data_set[ct,hregion[0],hregion[1],hregion[2]] = shared_arr[2,regions[ONE][1],regions[ONE][2],regions[ONE][3]]
            comm.Barrier()
            np.copyto(shared_arr[0,wr[1],wr[2],wr[3]],shared_arr[2,wr[1],wr[2],wr[3]])
            ct+=1
        np.copyto(shared_arr[1,wr[1],wr[2],wr[3]],shared_arr[2,wr[1],wr[2],wr[3]])
        shared_arr[2,wr[1],wr[2],wr[3]] = 0
        reg_edge_comm(shared_arr,OPS,brs,regions[ONE])
        comm.Barrier()
        # if rank == master_rank:
        #     pm(shared_arr,1)
        #     input()
        # comm.Barrier()
    #Final barrier
    comm.Barrier()
    #CUDA clean up - One of the last steps
    if gpu_rank[0]:
        cuda_context.pop()
    comm.Barrier()
    #Timer
    stop = timer.time()
    exec_time = abs(stop-start)
    exec_time = comm.gather(exec_time)
    if rank == master_rank:
        avg_time = sum(exec_time)/num_ranks
        hdf_time[0] = avg_time
    comm.Barrier()
    #Closing file
    hdf5_file.close()
    #For testing only
    if rank == master_rank:
        return avg_time

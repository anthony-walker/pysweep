
#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs

import sys, os, h5py, math
import numpy as np
from itertools import cycle, product

#CUDA Imports
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#Dsweep imports
from decomposition import create_blocks, node_split, read_input_file

from block import *
from source import *

#MPI imports
from mpi4py import MPI
#Multi-processing imports
from concurrent import futures

#GPU Utility Imports
import GPUtil


import importlib.util
#Testing and Debugging
import warnings
import time as timer
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS

def dsweep_engine():
    # arr0,gargs,swargs,filename ="results",exid=[],dType=np.dtype('float32')
    """Use this function to perform swept rule
    args:
    arr0 -  3D numpy array of initial conditions (v (variables), x,y)
    gargs:   (global args)
        The first three variables must be time arguments (t0,tf,dt).
        Subsequent variables can be anything passed to the source code in the
        "set_globals" function
    swargs: (Swept args)
        TSO - order of the time scheme
        OPS - operating points to remove from each side for a function step
            (e.g. a 5 point stencil would result in OPS=2).
        BS - an integer which represents the x and y dimensions of the gpu block size
        AF -  the GPU affinity (GPU work/CPU work)/TotalWork
        GS - GPU source code file
        CS - CPU source code file
    dType: data type for the run (e.g. np.dtype("float32")).
    filename: Name of the output file excluding hdf5
    exid: GPU ids to exclude from the calculation.
    """
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    rank = comm.Get_rank()  #current rank
    all_ranks = comm.allgather(rank) #All ranks in simulation
    #Getting input data
    global OPS,TSO,AF
    arr0,gargs,swargs,GS,CS,filename,exid,dType = read_input_file(comm)
    TSO,OPS,BS,AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    t0,tf,dt = gargs[:3]
    BS = (BS,BS,1)
    comm.Barrier()
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2
    TOPS = TWO*OPS

    #Create individual node comm
    nodes_processors = comm.allgather((rank,processor))
    processors = tuple(zip(*nodes_processors))[1]
    node_ranks = [n for n,p in nodes_processors if p==processor]
    node_master = node_ranks[0]

    #Create cluster comm
    cluster_ranks = list(set(comm.allgather(node_master)))
    cluster_master = cluster_ranks[0]
    cluster_group = comm.group.Incl(cluster_ranks)
    cluster_comm = comm.Create_group(cluster_group)

    #CPU Core information
    processors = set(processors)
    total_num_cpus = len(processors)
    assert comm.Get_size()%total_num_cpus==0,'Number of process requirements are not met'
    num_cores = os.cpu_count()
    total_num_cores = num_cores*total_num_cpus #Assumes all nodes have the same number of cores in CPU

    if node_master == rank:
        if AF>0:
            gpu_rank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=exid,limit=1e8) #getting devices by load
            gpu_rank = [(True,id) for id in gpu_rank]
            num_gpus = len(gpu_rank)
        else:
            gpu_rank = []
            num_gpus = 0
        num_cores -= num_gpus
        ranks_to_remove = list()
        #Finding ranks to remove - keeps 1 for GPU operations
        while len(node_ranks) > num_gpus+1:
            ranks_to_remove.append(node_ranks.pop())
        total_num_gpus = np.sum(cluster_comm.allgather(num_gpus))
        #Testing ranks and number of gpus to ensure simulation is viable
        assert total_num_gpus < comm.Get_size() if AF < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
        assert total_num_gpus > 0 if AF > 0 else True, "There are no avaliable GPUs"
        #Get total number of blocks
        NB = np.prod(arr0.shape[1:])/np.prod(BS)
        assert (NB).is_integer(), "Provided array dimensions is not divisible by the specified block size."
        num_block_rows = arr0.shape[1]/BS[0]
        #This function splits the rows among the nodes by affinity
        node_ids,node_row_list,rows_per_gpu = node_split(cluster_comm,rank,cluster_master,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus)
        node_row_list = [int(x) for x in node_row_list]
        #Getting nodes in front and behind current node
        nidx = node_ids.index(rank)
        node_rows = node_row_list[nidx]
        bidx = node_ids.index(-1) if  - 1 >= 0 else len(node_ids)-1
        bnode = (node_ids[bidx],node_row_list[bidx])
        fidx = node_ids.index(+1) if  + 1 < len(node_ids) else 0
        fnode = (node_ids[fidx],node_row_list[fidx])
    else:
        node_row_list = None
        node_rows=None
        nidx = None
        ranks_to_remove = None

    #----------------------__Removing Unwanted MPI Processes------------------------#
    [all_ranks.remove(x) for x in set([x[0] for x in comm.allgather(ranks_to_remove) if x])]
    comm.Barrier()
    #Remaking comms
    if rank in all_ranks:
        #New Global comm
        ncg = comm.group.Incl(all_ranks)
        comm = comm.Create_group(ncg)
        #New node comm
        node_group = comm.group.Incl(node_ranks)
        node_comm = comm.Create_group(node_group)
    else: #Ending unnecs
        MPI.Finalize()
        exit(0)
    comm.Barrier()
    #Checking to ensure that there are enough
    assert total_num_gpus >= comm.Get_size() if AF == 1 else True,"Not enough GPUs for ranks"
    #Broad casting to node
    node_rows = node_comm.bcast(node_rows)
    nidx = node_comm.bcast(nidx)
    node_row_list = node_comm.bcast(node_row_list)

    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Splits for shared array
    SPLITX = int(BS[ZERO]/TWO)+OPS   #Split computation shift - add OPS
    SPLITY = int(BS[ONE]/TWO)+OPS   #Split computation shift
    global up_sets, down_sets, oct_sets, x_sets, y_sets
    up_sets = create_up_sets(BS,OPS)
    down_sets = create_down_sets(BS,OPS)[:-1]
    oct_sets = down_sets+up_sets[1:]
    MPSS = len(up_sets)
    MOSS = len(oct_sets)
    bridge_sets, bridge_slices = create_bridge_sets(BS,OPS,MPSS)
    x_sets = bridge_sets[0]
    y_sets = bridge_sets[1]
    IEP = 1 if MPSS%2==0 else 0 #This accounts for even or odd MPSS

    #----------------Data Input setup -------------------------#
    time_steps = int((tf-t0)/dt)  #Number of time steps
    #Global swept step
    MGST = int(TSO*(time_steps)/(MOSS)-1)    #THIS ASSUMES THAT time_steps > MOSS
    time_steps = int((MGST*(MOSS)/TSO)+MPSS) #Updating time steps

    #-----------------------SHARED ARRAY CREATION----------------------#
    #Create shared process array for data transfer  - TWO is added to shared shaped for IC and First Step
    shared_shape = (MOSS+TSO+ONE,arr0.shape[0],int(node_rows*BS[0])+2*OPS,arr0.shape[2])
    global sarr
    sarr = create_CPU_sarray(node_comm,shared_shape,dType,np.prod(shared_shape)*dType.itemsize)
    #Filling shared array
    gsc =slice(int(BS[0]*np.sum(node_row_list[:nidx])),int(BS[0]*(np.sum(node_row_list[:nidx])+node_rows)),1)
    sarr[0,:,OPS:-OPS,:] =  arr0[:,gsc,:]
    sarr[0,:,:OPS,:] =  arr0[:,np.arange(gsc.start-OPS,gsc.start),:]
    sarr[0,:,-OPS:,:] =  arr0[:,np.arange(gsc.start,gsc.start+OPS),:]
    ssb = np.zeros((2,arr0.shape[ZERO],BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes

    #Creating blocks to be solved
    if rank==node_master:
        gpu_blocks,cpu_blocks = create_blocks(shared_shape,rows_per_gpu,BS,num_gpus,OPS)
        gpu_ranks = node_ranks[:num_gpus]
        cpu_ranks = node_ranks[num_gpus:]
        blocks = np.array_split(gpu_blocks,num_gpus) if gpu_blocks else gpu_blocks
        blocks.append(cpu_blocks) if cpu_blocks else None
        gpu_rank += [(False,None) for i in range(len(cpu_ranks))]
        node_data = zip(blocks,gpu_rank)
    else:
        node_data = None
    blocks,gpu_rank =  node_comm.scatter(node_data)
    GRB = gpu_rank[0]
    global SM
    #Operations specifically for GPus and CPUs
    if GRB:
        # Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank[ONE])
        cuda_context = cuda_device.make_context()
        blocks = tuple(blocks[0])
        larr = np.copy(sarr[blocks])
        # Creating local GPU array with split
        GRD = (int((larr.shape[TWO])/BS[ZERO]),int((larr.shape[3])/BS[ONE]))   #Grid size
        #Creating constants
        NV = larr.shape[ONE]
        SGIDS = (BS[ZERO]+TWO*OPS)*(BS[ONE]+TWO*OPS)
        STS = SGIDS*NV #Shared time shift
        VARS =  larr.shape[TWO]*larr.shape[3]
        TIMES = VARS*NV
        const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})
        #Building CUDA source code
        SM = build_gpu_source(GS)
        swept_constant_copy(SM,const_dict)
        cpu_SM = build_cpu_source(CS)   #Building cpu source for set_globals
        cpu_SM.set_globals(GRB,SM,*gargs)
        del cpu_SM
        del larr
        mpi_pool = None
    else:
        SM = build_cpu_source(CS) #Building Python source code
        SM.set_globals(GRB,SM,*gargs)
        GRD = None
        mpi_pool = futures.ProcessPoolExecutor(os.cpu_count()-node_comm.Get_size()+1)

    #------------------------------HDF5 File------------------------------------------#
    # filename+=".hdf5"
    # hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=comm)
    # hdf_bs = hdf5_file.create_dataset("BS",(len(BS),))
    # hdf_bs[:] = BS[:]
    # hdf_as = hdf5_file.create_dataset("array_size",(len(arr0.shape)+ONE,))
    # hdf_as[:] = (time_steps,)+arr0.shape
    # hdf_aff = hdf5_file.create_dataset("AF",(ONE,))
    # hdf_aff[ZERO] = AF
    # hdf_time = hdf5_file.create_dataset("time",(1,))
    # hdf5_data_set = hdf5_file.create_dataset("data",(time_steps+ONE,arr0.shape[ZERO],arr0.shape[ONE],arr0.shape[TWO]),dtype=dType)
    # hregion = (WR[1],slice(WR[2].start-OPS,WR[2].stop-OPS,1),slice(WR[3].start-OPS,WR[3].stop-OPS,1))
    # hdf5_data_set[0,hregion[0],hregion[1],hregion[2]] = sarr[TSO-ONE,WR[1],WR[2],WR[3]]
    cwt = 1 #Current write time
    global gts
    gts = 0  #Counter for writing on the appropriate step
    comm.Barrier() #Ensure all processes are prepared to solve
    # #-------------------------------SWEPT RULE---------------------------------------------#
    pargs = (SM,GRB,BS,GRD,OPS,TSO,ssb) #Passed arguments to the swept functions
    #-------------------------------FIRST PYRAMID-------------------------------------------#
    UpPrism(sarr,blocks,up_sets,x_sets,gts,pargs,mpi_pool)

    #Pop Cuda Contexts
    if GRB:
        cuda_context.pop()
    else:
        mpi_pool.shutdown()
    comm.Barrier()


def UpPrism(sarr,blocks,up_sets,x_sets,gts,pargs,mpi_pool):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        pass
        # Getting GPU Function
        # arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        # gpu_fcn = SM.get_function("UpPyramid")
        # gpu_fcn(cuda.InOut(arr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        # cuda.Context.synchronize()
    else:   #CPUs do this
        mpi_pool.map(CPU_UpPrism,blocks)


def CPU_UpPrism(block):
    """Use this function to build the Up Pyramid."""
    print('UpPrism')
    #UpPyramid of Swept Step
    for ts,swept_set in enumerate(up_sets,start=TSO-1):
        #Calculating Step
        block = SM.step(sarr[block],swept_sets,ts,gts)
        gts+=1
    #X-Bridge - NEED TO SHIFT THIS
    for ts,swept_set in enumerate(x_sets,start=TSO):
        #Calculating Step
        block = SM.step(sarr[block],swept_sets,ts,gts)
        gts+=1
    return block

#Statement to execute dsweep
dsweep_engine()
#Statement to finalize MPI processes
MPI.Finalize()

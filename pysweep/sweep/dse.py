#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs
import sys, os, h5py, math, GPUtil, importlib.util, ctypes, time
from itertools import cycle, product, count
#CUDA Imports
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#Dsweep imports
from dcore import dcore,decomp,functions,sgs
from ccore import source, printer
#MPI imports
from mpi4py import MPI
#Multi-processing imports
import multiprocessing as mp
import numpy as np


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
    #Setting global variables
    sgs.init_globals()
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    rank = comm.Get_rank()  #current rank
    all_ranks = comm.allgather(rank) #All ranks in simulation
    #Getting input data
    arr0,gargs,swargs,GS,CS,filename,exid,dType = decomp.read_input_file(comm)
    TSO,OPS,BS,AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    sgs.TSO,sgs.OPS,sgs.BS,sgs.AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    t0,tf,dt = gargs[:3]
    assert BS%(2*OPS)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
    BS = (BS,BS,1)
    comm.Barrier()
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
    #------------------------Getting Avaliable Architecture and Decomposing Data-------------------------------#
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
        node_ids,node_row_list,rows_per_gpu = decomp.node_split(cluster_comm,rank,cluster_master,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus)
        node_row_list = [int(x) for x in node_row_list]
        #Getting nodes in front and behind current node
        nidx = node_ids.index(rank)
        node_rows = node_row_list[nidx]
        bidx = node_ids.index(-1) if  - 1 >= 0 else len(node_ids)-1
        bnode = (node_ids[bidx],node_row_list[bidx])
        fidx = node_ids.index(+1) if  + 1 < len(node_ids) else 0
        fnode = (node_ids[fidx],node_row_list[fidx])
    else:
        node_row_list,node_rows,nidx,ranks_to_remove,rows_per_gpu,num_gpus,gpu_rank,total_num_gpus = None,None,None,None,None,None,None,None
    #----------------------__Removing Unwanted MPI Processes------------------------#
    node_rows,nidx,node_row_list,node_comm = dcore.mpi_destruction(rank,node_ranks,node_rows,comm,ranks_to_remove,all_ranks,nidx,node_row_list,total_num_gpus,AF)
    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Splits for shared array
    SPLITX = int(BS[ZERO]/TWO)+OPS   #Split computation shift - add OPS
    SPLITY = int(BS[ONE]/TWO)+OPS   #Split computation shift
    MPSS = int(BS[0]/(2*OPS)-1)
    MOSS = 2*MPSS
    time_steps = int((tf-t0)/dt)  #Number of time steps
    MGST = int(TSO*(time_steps)/(MOSS)-1)  #Global swept step  #THIS ASSUMES THAT time_steps > MOSS
    time_steps = int((MGST*(MOSS)/TSO)+MPSS) #Updating time steps
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (MOSS+TSO+ONE,arr0.shape[0],int(node_rows*BS[0]),arr0.shape[2])
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,np.zeros(shared_shape).nbytes)
    ssb = np.zeros((2,arr0.shape[ZERO],BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
    #Filling shared array
    gsc =slice(int(BS[0]*np.sum(node_row_list[:nidx])),int(BS[0]*(np.sum(node_row_list[:nidx])+node_rows)),1)
    sarr[TSO-ONE,:,:,:] =  arr0[:,gsc,:]
    #----------------Creating and disseminating blocks -------------------------#
    GRB,blocks,total_cpu_block,gpu_rank = dcore.block_dissem(rank,node_master,shared_shape,rows_per_gpu,BS,num_gpus,OPS,node_ranks,gpu_rank,node_comm)
    #------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and Context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank[ONE])
        cuda_context = cuda_device.make_context()
        blocks,block_shape,GRD,garr = dcore.gpu_core(blocks,BS,OPS,GS,CS,gargs,GRB,MPSS,MOSS,TSO)
        mpi_pool,carr,up_sets,down_sets,oct_sets,x_sets,y_sets = None,None,None,None,None,None,None
    else:
        GRD,block_shape,garr = None,None,None
        blocks = dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,CS,GRB,gargs,MPSS,total_cpu_block)
        mpi_pool = mp.Pool(os.cpu_count()-node_comm.Get_size()+1)
    #------------------------------HDF5 File------------------------------------------#
    hdf5_file = dcore.make_hdf5(filename,cluster_master,comm,rank,BS,arr0,time_steps,AF,dType)
    cwt = 1 #Current write time
    comm.Barrier() #Ensure all processes are prepared to solve
    #-------------------------------SWEPT RULE---------------------------------------------#
    pargs = (sgs.SM,GRB,BS,GRD,OPS,TSO,ssb) #Passed arguments to the swept functions
    #-------------------------------FIRST PYRAMID-------------------------------------------#
    functions.FirstPrism(sarr,garr,blocks,sgs.gts,pargs,mpi_pool,total_cpu_block)
    node_comm.Barrier()
    if rank == node_master:
        for i in range(2,4,1):
            print('-----------------------------------------')
            printer.pm(sarr,i)

    #Clean Up - Pop Cuda Contexts and Close Pool
    if GRB:
        cuda_context.pop()
    # else:
    #     mpi_pool.shutdown()
    comm.Barrier()
    hdf5_file.close()




#Statement to execute dsweep
dsweep_engine()
#Statement to finalize MPI processes
MPI.Finalize()

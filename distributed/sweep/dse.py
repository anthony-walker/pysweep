#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs
import sys, os, h5py, math, GPUtil, time
from itertools import cycle
#CUDA Imports
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
#Dsweep imports
from dcore import dcore,decomp,functions,sgs
from ccore import source, printer
#MPI imports
from mpi4py import MPI
#Multi-processing imports
import multiprocessing as mp
import numpy as np



def dsweep_engine():
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
    filename: Name of the output file excluding hdf5
    exid: GPU ids to exclude from the calculation.
    """
    #Starting timer
    start = time.time()
    #Setting global variables
    sgs.init_globals()
    #Local Constants
    ZERO = 0
    QUARTER = 0.25
    HALF = 0.5
    ONE = 1
    TWO = 2
    comm = MPI.COMM_WORLD
    #------------------INPUT DATA SETUP-------------------------$
    arr0,gargs,swargs,GS,CS,filename,exid,dType = decomp.read_input_file(comm)
    TSO,OPS,BS,AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    sgs.TSO,sgs.OPS,sgs.BS,sgs.AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    t0,tf,dt = gargs[:3]
    assert BS%(2*OPS)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
    BS = (BS,BS,1)
    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Splits for shared array
    SPLITX = int(BS[ZERO]/TWO)   #Split computation shift - add OPS
    SPLITY = int(BS[ONE]/TWO)   #Split computation shift
    MPSS = int(BS[0]/(2*OPS)-1)
    sgs.MPSS = MPSS
    MOSS = 2*MPSS
    time_steps = int((tf-t0)/dt)  #Number of time steps
    MGST = int(TSO*(time_steps-MPSS)/(MPSS)+1)  #Global swept step  #THIS ASSUMES THAT time_steps > MOSS
    time_steps = int(MPSS*(MGST+1)/TSO+1) #Number of time steps - Add 1 for initial conditions
    #-------------MPI SETUP----------------------------#
    processor = MPI.Get_processor_name()
    rank = comm.Get_rank()  #current rank
    all_ranks = comm.allgather(rank) #All ranks in simulation
    #Create individual node comm
    nodes_processors = comm.allgather((rank,processor))
    processors = tuple(zip(*nodes_processors))[1]
    node_ranks = [n for n,p in nodes_processors if p==processor]
    processors = set(processors)
    node_group = comm.group.Incl(node_ranks)
    node_comm = comm.Create_group(node_group)
    node_master = node_ranks[0]
    NMB = rank == node_master
    #Create cluster comm
    cluster_ranks = list(set(comm.allgather(node_master)))
    cluster_master = cluster_ranks[0]
    cluster_group = comm.group.Incl(cluster_ranks)
    cluster_comm = comm.Create_group(cluster_group)

    #CPU Core information
    total_num_cpus = len(processors)
    num_cpus = 1 #Each rank will always have 1 cpu
    num_cores = os.cpu_count()
    total_num_cores = num_cores*total_num_cpus #Assumes all nodes have the same number of cores in CPU
    if NMB:
        #Giving each node an id
        if cluster_master == rank:
            node_id = np.arange(1,cluster_comm.Get_size()+1,1,dtype=np.intc)
        else:
            node_id = None
        node_id = cluster_comm.scatter(node_id)
        #Getting GPU information
        gpu_rank,total_num_gpus, num_gpus, node_info, comranks, GNR,CNR = dcore.get_gpu_info(rank,cluster_master,node_id,cluster_comm,AF,BS,exid,processors,node_comm.Get_size(),arr0.shape)
        ranks_to_remove = dcore.find_remove_ranks(node_ranks,AF,num_gpus)
        [gpu_rank.append(None) for i in range(len(node_ranks)-len(gpu_rank))]
        #Testing ranks and number of gpus to ensure simulation is viable
        assert total_num_gpus < comm.Get_size() if AF < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
        assert total_num_gpus > 0 if AF > 0 else True, "There are no avaliable GPUs"
        assert total_num_gpus <= GNR if AF > 0 else True, "Not enough rows for the number of GPUS, added more GPU rows, increase affinity, or exclude GPUs."
    else:
        total_num_gpus,node_info,gpu_rank,node_id,num_gpus,comranks = None,None,None,None,None,None
        ranks_to_remove = []
    #Broadcasting gpu information
    total_num_gpus = comm.bcast(total_num_gpus)
    node_ranks = node_comm.bcast(node_ranks)
    node_info = node_comm.bcast(node_info)
    #----------------------__Removing Unwanted MPI Processes------------------------#
    node_comm,comm = dcore.mpi_destruction(rank,node_ranks,comm,ranks_to_remove,all_ranks)
    gpu_rank,blocks = decomp.nsplit(rank,node_master,node_comm,num_gpus,node_info,BS,arr0.shape,gpu_rank)
    # Checking to ensure that there are enough
    assert total_num_gpus >= node_comm.Get_size() if AF == 1 else True,"Not enough GPUs for ranks"
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (MOSS+TSO+ONE,arr0.shape[0],int(sum(node_info[2:])*BS[0]),arr0.shape[2])
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,np.zeros(shared_shape).nbytes)
    ssb = np.zeros((2,arr0.shape[ZERO],BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
    #Filling shared array
    if NMB:
        gsc = (slice(0,arr0.shape[1],1),slice(int(node_info[0]*BS[0]),int(node_info[1]*BS[0]),1),slice(0,arr0.shape[2],1))
        sarr[TSO-ONE,:,:,:] =  arr0[gsc]
    else:
        gsc = None
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in shared_shape]
    blocks = (bsls[0],bsls[1],blocks,bsls[3])
    GRB = True if gpu_rank is not None else False
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        block_shape,GRD,garr = dcore.gpu_core(blocks,BS,OPS,GS,CS,gargs,GRB,MPSS,MOSS,TSO)
        mpi_pool,carr,up_sets,down_sets,oct_sets,x_sets,y_sets,total_cpu_block = None,None,None,None,None,None,None,None
    else:
        GRD,block_shape,garr = None,None,None
        blocks,total_cpu_block = dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,CS,GRB,gargs,MPSS)
        pool_size = min(len(blocks), os.cpu_count()-node_comm.Get_size()+1)
        mpi_pool = mp.Pool(pool_size)
    # ------------------------------HDF5 File------------------------------------------#
    hdf5_file, hdf5_data,hdf_time = dcore.make_hdf5(filename,cluster_master,comm,rank,BS,arr0,time_steps,AF,dType,gargs)
    comm.Barrier() #Ensure all processes are prepared to solve
    # -------------------------------SWEPT RULE---------------------------------------------#
    pargs = (sgs.SM,GRB,BS,GRD,OPS,TSO,ssb) #Passed arguments to the swept functions
    node_comm.Barrier()
    # -------------------------------FIRST PRISM AND COMMUNICATION-------------------------------------------#
    functions.FirstPrism(sarr,garr,blocks,sgs.gts,pargs,mpi_pool,total_cpu_block)
    node_comm.Barrier()
    functions.first_forward(NMB,GRB,node_comm,cluster_comm,comranks,sarr,SPLITX,total_cpu_block)
    cwt = 1
    # -------------------------------SWEPT LOOP--------------------------------------------#
    step = cycle([functions.send_backward,functions.send_forward])
    for i in range(MGST):
        functions.UpPrism(sarr,garr,blocks,sgs.gts,pargs,mpi_pool,total_cpu_block)
        node_comm.Barrier()
        cwt = next(step)(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,SPLITX,total_cpu_block)
    #Do LastPrism Here then Write all of the remaining data
    functions.LastPrism(sarr,garr,blocks,sgs.gts,pargs,mpi_pool,total_cpu_block)
    node_comm.Barrier()
    next(step)(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,SPLITX,total_cpu_block)
    # Clean Up - Pop Cuda Contexts and Close Pool
    if GRB:
        cuda_context.pop()
    comm.Barrier()
    stop = time.time()
    hdf_time[0] = stop-start
    hdf5_file.close()
    #Removing input file.
    if rank==cluster_master:
        os.system("rm "+"input_file.hdf5") #remove input file
        if os.path.isfile('log.hdf5'):
            log_file = h5py.File('log.hdf5','a')
            shape = log_file['time'].shape[0]
            log_file['time'].resize((log_file['time'].shape[0]+1),axis=0)
            log_file['time'][shape]=stop-start
            log_file['type'].resize((log_file['type'].shape[0]+1),axis=0)
            log_file['type'][shape]=ord('s')
            log_file['args'].resize((log_file['args'].shape[0]+1),axis=0)
            log_file['args'][shape]=gargs
            log_file.close()
        else:
            log_file = h5py.File('log.hdf5','w')
            log_file.create_dataset('time',(1,),data=(stop-start),chunks=True,maxshape=(None,))
            log_file.create_dataset('type',(1,),data=(ord('s')),chunks=True,maxshape=(None,))
            log_file.create_dataset('args',(1,)+np.shape(gargs),data=gargs,chunks=True,maxshape=(None,)+np.shape(gargs))
            log_file.close()


#Statement to execute dsweep
dsweep_engine()

#Statement to finalize MPI processes
MPI.Finalize()

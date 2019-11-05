
#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs
import sys, h5py, GPUtil, time
import numpy as np
from mpi4py import MPI

def ddecomp(arr0,gargs,swargs,filename ="results",exid=[],dType='float32'):
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
    dType: a string data type which will be entered into numpy to obtain a datatype object
    """
    start = time.time()

    AF = swargs[3]
    swnames = ['TSO','OPS','BS','AF']
    #Getting GPU info
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    proc_mult = 1
    rank = comm.Get_rank()  #current rank
    master_rank = 0
    #Making sure arr0 has correct data type
    arr0 = arr0.astype(dType)
    #Create input file
    ifn = "input_file.hdf5"
    hdf5_file = h5py.File(ifn, 'w', driver='mpio', comm=comm)
    hdf5_file.create_dataset("arr0",arr0.shape,data=arr0)
    hdf5_file.create_dataset("swargs",(len(swargs[:-2]),),data=swargs[:-2])
    hdf5_file.create_dataset("gargs",(len(gargs),),data=gargs[:])
    GS = [ord(ch) for ch in swargs[4]]
    CS = [ord(ch) for ch in swargs[5]]
    FN = [ord(ch) for ch in filename]
    DT = [ord(ch) for ch in dType]
    hdf5_file.create_dataset("GS",(len(GS),),data=GS)
    hdf5_file.create_dataset("CS",(len(CS),),data=CS)
    hdf5_file.create_dataset("exid",(len(exid),),data=exid)
    hdf5_file.create_dataset("filename",(len(FN),),data=FN)
    hdf5_file.create_dataset("dType",(len(DT),),data=DT)
    comm.Barrier()
    hdf5_file.close()
    #Getting GPUs if affinity is greater than 0
    if AF>0:
        gpu_rank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=exid,limit=1e8) #getting devices by load
        num_gpus = len(gpu_rank)
    else:
        gpu_rank = []
        num_gpus = 0
    node_gpus_list = comm.allgather(num_gpus)
    if rank == master_rank:
        #Multipler for Number of processes to add
        proc_mult += max(node_gpus_list)
    proc_mult = comm.bcast(proc_mult,root=master_rank)
    comm.Barrier()
    num_procs = int(proc_mult*comm.Get_size())
    # Spawning MPI processes
    if rank == master_rank:
        MPI.COMM_SELF.Spawn(sys.executable,args=['./pysweep/distributed/sweep/dece.py'],maxprocs=num_procs)
    else:
        MPI.Finalize()
    stop = time.time()
    return stop-start

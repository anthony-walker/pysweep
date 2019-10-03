
#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src/sweep")
#Writing imports
import h5py

#DataStructure and Math imports
import math
import numpy as np
import ctypes
from collections import deque
from itertools import cycle
from itertools import product

#CUDA Imports
import pycuda.driver as cuda
import pycuda.autoinit  #Cor debugging only
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
from .pysweep_lambda import sweep_lambda
from .pysweep_functions import *
from .pysweep_decomposition import *
from pysweep_block import *
from .pysweep_regions import *
from .pysweep_source import *
import importlib.util
#Testing and Debugging
import warnings
import time as timer
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS


def dsweep(arr0,gargs,swargs,filename ="results",exid=[]):
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
    #Unpacking arguments
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
    nodes = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    master_node = ZERO #master rank
    total_num_cpus = nodes.Get_size() #number of ranks
    node = nodes.Get_rank()  #current rank
    num_cores = pool_size = mp.cpu_count() - 8
    total_num_cores = num_cores*total_num_cpus #Assumes all nodes have the same number of cores in CPU
    proc_pool = mp.Pool(pool_size)

    if node == 0:
        exid = [1]
    else:
        exid = [0]

    #Getting GPUs if affinity is greater than 1
    if AF>0:
        gpu_node = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=exid,limit=1e8) #getting devices by load
        gpu_node = [(True,id) for id in gpu_node]
        num_gpus = len(gpu_node)
    else:
        gpu_node = []
        num_gpus = 0
    num_cores -= num_gpus
    #Get total number of GPUs
    total_num_gpus = np.sum(nodes.allgather(num_gpus))
    #Get total number of blocks
    NB = np.prod(arr0.shape[1:])/np.prod(BS)
    assert (NB).is_integer()
    num_block_rows = arr0.shape[1]/BS[0]
    #This function splits the rows among the nodes by affinity
    node_ids,node_row_list,rows_per_gpu = node_split(nodes,node,master_node,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus)
    node_row_list = [int(x) for x in node_row_list]
    #Getting nodes in front and behind current node
    nidx = node_ids.index(node)
    node_rows = node_row_list[nidx]
    bidx = node_ids.index(node-1) if node - 1 >= 0 else len(node_ids)-1
    bnode = (node_ids[bidx],node_row_list[bidx])
    fidx = node_ids.index(node+1) if node + 1 < len(node_ids) else 0
    fnode = (node_ids[fidx],node_row_list[fidx])

    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Splits for shared array
    SPX = int(BS[ZERO]/TWO)+OPS   #Split computation shift - add OPS
    # SPY = int(BS[ONE]/TWO)+OPS   #Split computation shift
    up_sets = create_up_sets(BS,OPS)
    down_sets = create_down_sets(BS,OPS)[:-1]
    oct_sets = down_sets+up_sets[1:]
    MPSS = len(up_sets)
    MOSS = len(oct_sets)
    bridge_sets, bridge_slices = create_bridge_sets(BS,OPS,MPSS)
    IEP = 1 if MPSS%2==0 else 0 #This accounts for even or odd MPSS

    #----------------Data Input setup -------------------------#
    time_steps = int((tf-t0)/dt)  #Number of time steps
    #Global swept step
    MGST = int(TSO*(time_steps)/(MOSS)-1)    #THIS ASSUMES THAT time_steps > MOSS
    time_steps = int((MGST*(MOSS)/TSO)+MPSS) #Updating time steps

    #-----------------------SHARED ARRAY CREATION----------------------#
    #Create shared process array for data transfer  - TWO is added to shared shaped for IC and First Step
    shared_shape = (MOSS+TSO+ONE,arr0.shape[0],int(node_rows*BS[0])+BS[0]+2*OPS,arr0.shape[2]+2*OPS)
    sarr_base = mp.Array(ctypes.c_float, int(np.prod(shared_shape)))
    sarr = np.ctypeslib.as_array(sarr_base.get_obj())
    sarr = sarr.reshape(shared_shape)

    #Filling shared array
<<<<<<< HEAD
    gsc =slice(int(BS[0]*np.sum(node_row_list[:nidx])),int(BS[0]*(np.sum(node_row_list[:nidx])+node_rows)),1)
    sarr[0,:,:,OPS:-OPS] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],:]
    sarr[0,:,:,:OPS] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],-OPS:]
    sarr[0,:,:,-OPS:] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],:OPS]
    create_blocks(shared_shape,node_rows,gpu_rows,BS,num_gpus,SPX,SPY,OPS)

def create_blocks(shared_shape,node_rows,gpu_rows,BS,num_gpus,spx,spy,ops):
    """Use this function to create blocks."""
    cpu_rows = node_rows-gpu_rows
    blocks = []
    for i in range(num_gpus):
        blocks += (slice(spx+BS[0]*node_rows*i,spx+BS[0]*node_rows*(i+1),1),slice(ops,shared_shape[3]-ops,1))
    cstart = BS[0]*node_rows*num_gpus+spx
    print(cstart)
    row_range = range(cstart,shared_shape[2],BS[0])
    print(row_range)
=======
    gsc = slice(int(BS[0]*np.sum(node_row_list[:nidx])),int(BS[0]*(np.sum(node_row_list[:nidx])+node_rows)),1)  #Slice of the global array
    sarr[0,:,:,OPS:-OPS] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],:]
    sarr[0,:,:,:OPS] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],-OPS:]
    sarr[0,:,:,-OPS:] =  arr0[:,np.arange(gsc.start-SPX,gsc.stop+SPX)%arr0.shape[1],:OPS]
>>>>>>> 407c1fbd609feb2db6cfefdbea6213307091554d

    #Creating blocks to be solved
    blocks = create_blocks(shared_shape,node_rows,rows_per_gpu,BS,num_gpus,SPX,OPS)
    #Synchronize nodes
    nodes.barrier()
    #Solve UpPyramid
    cpu_fcn = sweep_lambda((UpPyramid,SM,isets,gts,TSO))

def UpPyramid(sarr,upsets,gts,pargs):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,CRS,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        # Getting GPU Function
        arr = np.ascontiguousarray(arr) #Ensure array is contiguous
        gpu_fcn = SM.get_function("UpPyramid")
        gpu_fcn(cuda.InOut(arr),np.int32(gts),grid=GRD, block=BS,shared=ssb)
        cuda.Context.synchronize()
    else:   #CPUs do this
        blocks = []
        for local_region in CRS:
            block = np.copy(arr[local_region])
            blocks.append(block)
        cpu_fcn = sweep_lambda((CPU_UpPyramid,SM,isets,gts,TSO))
        blocks = list(map(cpu_fcn,blocks))
        arr = rebuild_blocks(arr,blocks,CRS,OPS)
    np.copyto(sarr[WR], arr[:,:,OPS:-OPS,OPS:-OPS])
    for br in BDR:
        np.copyto(sarr[br[0],br[1],br[4],br[5]],arr[br[0],br[1],br[2],br[3]])



def create_blocks(shared_shape,node_rows,rows_per_gpu,BS,num_gpus,spx,ops):
    """Use this function to create blocks."""
    cpu_rows = node_rows-rows_per_gpu
    blocks = []
    for i in range(num_gpus):
        blocks.append((slice(int(spx+BS[0]*rows_per_gpu*i),int(spx+BS[0]*rows_per_gpu*(i+1)),1),slice(ops,shared_shape[3]-ops,1)))
    cstart = blocks[-1][0].stop
    row_range = np.arange(cstart,shared_shape[2]-spx,BS[0],dtype=np.intc)
    column_range = np.arange(ops,shared_shape[3]-int(BS[1]/2)+ops,BS[1],dtype=np.intc)
    blocks += [(slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)]
    return blocks



def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%1.1f"%si+", ")
        sys.stdout.write("]\n")

def node_split(nodes,node,master_node,num_block_rows,AF,total_num_cores,num_cores,total_num_gpus,num_gpus):
    """Use this function to split data amongst nodes."""
    #Assert that the total number of blocks is an integer
    gpu_rows = np.ceil(num_block_rows*AF)
    rows_per_gpu = gpu_rows/total_num_gpus if total_num_gpus > 0 else 0
    cpu_rows = num_block_rows-gpu_rows
    rows_per_core = np.ceil(cpu_rows/total_num_cores) if num_cores > 0 else 0
    #Estimate number of rows per node
    node_rows = rows_per_gpu*num_gpus+rows_per_core*num_cores
    nlst = None
    #Number of row correction
    tnr = nodes.gather((node,node_rows))
    cores = nodes.gather(num_cores)
    if node == master_node:
        nlst = [x for x,y in tnr]
        tnr = [y for x,y in tnr]
        total_rows = np.sum(tnr)
        cis = cycle(tuple(np.zeros(nodes.Get_size()-1))+(1,))
        ncyc = cycle(range(0,nodes.Get_size(),1))
        #If total rows are greater, reduce
        min_cores = min(cores)
        while total_rows > num_block_rows:
            curr = next(ncyc)
            if min_cores >= cores[curr]:
                tnr[curr]-=1
                total_rows-=1
            min_cores += next(cis)
        #If total rows are greater, add
        max_cores = max(cores)
        while total_rows < num_block_rows:
            curr = next(ncyc)
            if max_cores <= cores[curr]:
                tnr[curr]+=1
                total_rows+=1
            max_cores -= next(cis)
    nodes.Barrier()
    return nodes.bcast((nlst,tnr),root=master_node)+(rows_per_gpu,)

def node_comm(nodes,fnode,cnode,bnode,sarr):
    """Use this function to communicate node data"""
    pass

    # ---------------Generating Source Modules----------------------------------#
    # if GRB:
    #     # Creating cuda device and Context
    #     cuda.init()
    #     cuda_device = cuda.Device(gpu_rank[ONE])
    #     cuda_context = cuda_device.make_context()
    #     #Creating local GPU array with split
    #     GRD = (int((larr.shape[TWO]-TOPS)/BS[ZERO]),int((larr.shape[3]-TOPS)/BS[ONE]))   #Grid size
    #     #Creating constants
    #     NV = larr.shape[ONE]
    #     SGIDS = (BS[ZERO]+TWO*OPS)*(BS[ONE]+TWO*OPS)
    #     STS = SGIDS*NV #Shared time shift
    #     VARS =  larr.shape[TWO]*larr.shape[3]
    #     TIMES = VARS*NV
    #     const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})
    #     #Building CUDA source code
    #     SM = build_gpu_source(GS)
    #     swept_constant_copy(SM,const_dict)
    #     cpu_SM = build_cpu_source(CS)   #Building cpu source for set_globals
    #     cpu_SM.set_globals(GRB,SM,*gargs)
    #     del cpu_SM
    #     CRS = None
    # else:
    #     SM = build_cpu_source(CS) #Building Python source code
    #     SM.set_globals(GRB,SM,*gargs)
    #     CRS = create_blocks_list(larr.shape,BS,OPS)
    #     GRD = None

    # #------------------------------HDF5 File------------------------------------------#
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
    # cwt = 1 #Current write time
    # wb = 0  #Counter for writing on the appropriate step
    # comm.Barrier() #Ensure all processes are prepared to solve
    # #-------------------------------SWEPT RULE---------------------------------------------#
    # pargs = (SM,GRB,BS,GRD,CRS,OPS,TSO,ssb) #Passed arguments to the swept functions
    # #-------------------------------FIRST PYRAMID-------------------------------------------#
    # UpPyramid(sarr,larr,WR,BDR,up_sets,wb,pargs) #THis modifies shared array
    # wb+=1
    # comm.Barrier()
    # #-------------------------------FIRST BRIDGE-------------------------------------------#
    # #Getting x and y arrays
    # xarr = np.copy(sarr[XR])
    # yarr = np.copy(sarr[YR])
    # comm.Barrier()  #Barrier after read
    # # Bridge Step
    # Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb,pargs) #THis modifies shared array
    # comm.Barrier()  #Solving Bridge Barrier
    # #------------------------------SWEPT LOOP-------------------------------#
    # #Getting next points for the local array
    # larr = np.copy(sarr[SRR])
    # comm.Barrier()
    # #Swept Octahedrons and Bridges
    # for GST in range(MGST):
    #     comm.Barrier()  #Read barrier for local array
    #     #-------------------------------FIRST OCTAHEDRON (NONSHIFT)-------------------------------------------#
    #     Octahedron(sarr,larr,SWR,SBDR,oct_sets,wb,pargs)
    #     comm.Barrier()  #Solving Barrier
    #     # Writing Step
    #     cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
    #     comm.Barrier()  #Write Barrier
    #     #Updating Boundary Conditions Step
    #     boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries
    #     comm.Barrier()
    #     #-------------------------------FIRST REVERSE BRIDGE-------------------------------------------#
    #     #Getting reverse x and y arrays
    #     xarr = np.copy(sarr[YR]) #Regions are purposely switched here
    #     yarr = np.copy(sarr[XR])
    #     comm.Barrier()  #Barrier after read
    #     #Reverse Bridge Step
    #     Bridge(sarr,xarr,yarr,wxts,wyts,bridge_sets,wb,pargs) #THis modifies shared array
    #     comm.Barrier()  #Solving Bridge Barrier
    #     #-------------------------------SECOND OCTAHEDRON (SHIFT)-------------------------------------------#
    #     #Getting next points for the local array
    #     larr = np.copy(sarr[RR])
    #     comm.Barrier()
    #     #Octahedron Step
    #     Octahedron(sarr,larr,WR,BDR,oct_sets,wb,pargs)
    #     comm.Barrier()  #Solving Barrier
    #     #Write step
    #     cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
    #     comm.Barrier()
    #     #Updating Boundary Conditions Step
    #     boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries
    #     comm.barrier()  #Barrier following data write
    #     #-------------------------------SECOND BRIDGE (NON-REVERSED)-------------------------------------------#
    #     #Getting x and y arrays
    #     xarr = np.copy(sarr[XR])
    #     yarr = np.copy(sarr[YR])
    #     comm.Barrier()  #Barrier after read
    #     #Bridge Step
    #     Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb,pargs) #THis modifies shared array
    #     comm.Barrier()
    #     #Getting next points for the local array
    #     larr = np.copy(sarr[SRR])
    # #Last read barrier for down pyramid
    # comm.Barrier()
    # #--------------------------------------DOWN PYRAMID------------------------#
    # DownPyramid(sarr,larr,SWR,SBDR,down_sets,wb,pargs)
    # comm.Barrier()
    # #Writing Step
    # hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
    # comm.Barrier()  #Write Barrier
    # # CUDA clean up - One of the last steps
    # if GRB:
    #     cuda_context.pop()
    # comm.Barrier()
    # #-------------------------------TIMING-------------------------------------#
    # stop = timer.time()
    # exec_time = abs(stop-start)
    # exec_time = comm.gather(exec_time)
    # if rank == master_rank:
    #     avg_time = sum(exec_time)/num_ranks
    #     hdf_time[0] = avg_time
    # comm.Barrier()
    # # Close file
    # hdf5_file.close()
    # comm.Barrier()
    # #This if for testing only
    # if rank == master_rank:
    #     return avg_time

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
import pycuda.autoinit
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
from dev import *
from pysweep_lambda import sweep_lambda
from pysweep_dev import *
from pysweep_functions import *

#Testing and Debugging
import platform
import time
sys.path.insert(0, '/home/walkanth/pysweep/src/equations')
import euler
import warnings
from euler import *
from pysweep_plot_tests import *
warnings.simplefilter("ignore") #THIS IGNORES WARNINGS
#--------------------Global Variables------------------------------#
#------------------Shared Memory Arrays----------------------------#
cpu_array = None
gpu_array = None
#------------------Architecture-----------------------------------------
gpu_id = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=1000) #getting devices by load
GPUtil.showUtilization()
dev_attrs = getDeviceAttrs(gpu_id[0],False)   #Getting device attributes
gpu = cuda.Device(gpu_id[0])    #Getting device with id via cuda
cores = mp.cpu_count()  #Getting number of cpu cores
root_cores = int(np.sqrt(cores))
#-----------------------------Functions----------------------------#
cpu_test_fcn = None
#-------------------------------Constants--------------------------#
SS = 0  #number of steps in Octahedron (Half Pyramid and DownPyramid)
#----------------------End Global Variables------------------------#

def sweep(arr0,targs,dx,dy,ops,block_size,kernel_source,affinity):
    """Use this function to perform swept rule
    args:
    arr0 -  3D numpy array of initial conditions (v (variables), x,y)
    targs - (t0,tf,dt)
    dx - spatial step on x axis
    dy - spatial step on y axis
    ops - number of atomic operations
    block_size - gpu block size (check your architecture requirements)
    affinity -  the GPU affinity (GPU work/CPU work)
    """
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    print("Rank: ",rank," Platform: ",platform.uname()[1])

    #----------------Reading In Source Code-------------------------------#
    source_code = source_code_read("./src/sweep/sweep.h")
    split_source_code = source_code.split("//!!(@#\n")
    source_code = split_source_code[0]+"\n"+source_code_read(kernel_source)+"\n"+split_source_code[1]
    source_mod = SourceModule(source_code)#,options=["--ptxas-options=-v"])

    #---------------Data Input setup -------------------------#
    time_steps = int((targs[1]-targs[0])/targs[2])+1  #Number of time steps +1 becuase initial time step
    arr = np.zeros((time_steps,)+arr0.shape,dtype=np.float32) #4D arr(t,v,x,y)
    arr[0] = arr0
    arr_shape = np.shape(arr)  #Shape of array
    #Grid size
    grid_size = (int(arr_shape[2]/block_size[0]),int(arr_shape[3]/block_size[1]))
    MSS,NV,SGIDS,VARS,TIMES,const_dict = constant_copy(source_mod,arr_shapegrid_size,block_size,ops)  #Copy data into constants and returning them for other use

    #------------------Dividing work based on the architecture----------------#
    if rank == master_rank:
        arr0s = rank_split(arr0,arr_shape[1:],num_ranks)

def create_shared_arrays():
    """Use this function to create shared memory arrays for node communication."""
    #----------------------------Creating shared arrays-------------------------#
    global cpu_array
    cpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    cpu_array = np.ctypeslib.as_array(cpu_array_base.get_obj())
    cpu_array = cpu_array.reshape(block_size)

    global gpu_array
    gpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    gpu_array = np.ctypeslib.as_array(gpu_array_base.get_obj())
    gpu_array = gpu_array.reshape(block_size)


def rank_split(arr0,num_size):
    """Use this function to equally split data among the ranks"""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(arr0,rank_size,axis=major_axis)

if __name__ == "__main__":
    # print("Starting execution.")
    dims = (4,int(128),int(128))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 0.1
    arr0[1,:,:] = 0.0
    arr0[2,:,:] = 0.2
    arr0[3,:,:] = 0.125

    #GPU Arguments
    block_size = (32,32,1)
    kernel = "./src/equations/euler.h"

    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.01
    targs = (t0,t_b,dt)

    #Space testing arguments
    dx = 0.1
    dy = 0.1
    ops = 2

    sweep(arr0,targs,dx,dy,order,block_size,kernel)

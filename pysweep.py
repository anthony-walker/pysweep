
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#System imports
import os
import sys

#DataStructure and Math imports
import math
import numpy as np
from collections import deque

#Parallel programming imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import GPUtil
import dev
from mpi4py import MPI
import multiprocessing as mp

def sweep(y0, dy, t0, t_b, dt,block_size,ops,devices,gpu_affinity):
    """Use this function to perform swept rule>"""
    #-------------MPI Set up----------------------------#
    comm = MPI.COMM_WORLD
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank

    #Set up of dimensions
    plane_shape = np.shape(y0)  #Shape of initial conditions

    #Get avaliable GPU's on a rank
    gpu_ids = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
    has_gpu = 0

    if gpu_ids:
        has_gpu = len(gpu_ids)
    has_gpu = comm.gather(has_gpu,root=0)
    #-------------------------Determining How Much Work----------------------#

    gpu_blocks,cpu_blocks,gpu_affinity = arch_work_blocks(plane_shape,block_size,gpu_affinity)
    GPUtil.showUtilization()
    dev_attrs = dev.getDeviceAttrs(0)


    #----------------------------CUDA TESTING------------------------------#
    #Array
    time_len = 2

    a =  np.ones((2,2,time_len,2),dtype=np.float32)
    a[:,:,0,:] = [2,3]

    #Getting cuda source
    source_code = source_code_read("./csrc/swept_source.cu")
    source_code += "\n"+source_code_read("./csrc/euler_source.cu")
    #Creating cuda source
    mod = SourceModule(source_code)

    #Constants
    dt = np.array([0.01,],dtype=np.float32)
    mss = np.array([100,],dtype=int)

    dt_ptr,_ = mod.get_global("dt")
    mss_ptr,_ = mod.get_global("mss")

    #Copying to GPU memory
    cuda.memcpy_htod(mss_ptr,mss)
    cuda.memcpy_htod(dt_ptr,dt)

    #Executing cuda source
    func = mod.get_function("sweep")
    func(cuda.InOut(a),grid=(1,1,1),block=(2,2,1))
    for x in a:
        for y in x:
            print("new time level")
            for t in y:
                print(t)
    #-------------------------END CUDA TESTING---------------------------#


def arch_work_blocks(plane_shape,block_size,gpu_affinity):
    """Use to determine the closest work split based on the specified gpu_affinity."""
    blocks_x = int(plane_shape[0]/block_size[0])
    blocks_y = int(plane_shape[1]/block_size[1])
    total_blocks = blocks_x*blocks_y
    gpu_blocks = round(total_blocks/(1+1/gpu_affinity))
    cpu_blocks = total_blocks-gpu_blocks
    return (gpu_blocks,cpu_blocks,gpu_blocks/cpu_blocks)

def arch_2D_split(part_dev_id,grid,block_size):
    """Use this function to split the array based on architecture."""
    try:
        #Axes
        t_ax = 0
        x_ax = 1
        y_ax = 2
        #Getting the partition device
        curr_dev = cuda.Device(part_dev_id)
        #Checking to max sure block size is not greater than allowable hardward dimensions
        b1 = (block_size[x_ax]>curr_dev.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X))
        b2 = (block_size[y_ax]>curr_dev.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y))
        b3 = (block_size[t_ax]>curr_dev.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z))
        if b1 or b2 or b3:
            raise Exception("Blocksize exceeds hardware specifications.")
        #Creating work element partition integers
        par_t = int(grid_shape[t_ax]/block_size[t_ax])
        par_x = int(grid_shape[x_ax]/block_size[x_ax])
        par_y = int(grid_shape[y_ax]/block_size[y_ax])
        # print("Par(x,y,t): ",par_x,", ",par_y,", ",par_t) #Printing partitions

        #Creating times list which includes elements at a specific time level
        time_parts = np.array_split(grid,par_t,axis=t_ax)
        times = list()
        for t_part in time_parts:
            time_level = dict()
            count = 0
            for x in np.array_split(t_part,par_x,axis=x_ax):
                for y in np.array_split(x,par_y,axis=y_ax):
                    time_level[count]=y
                    count+=1
            times.append(time_level)
        return times
    except Exception as e:
        print(e)

def source_code_read(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source


if __name__ == "__main__":
    dims = (5*256,5*256,4)
    y0 = np.zeros(dims)+1
    dy = [0.1,0.1]
    t0 = 0
    t_b = 1
    dt = 0.001
    order = 2
    block_size = (256,256,1)
    sweep(y0,dy,t0,t_b,dt,block_size,2,0,40)

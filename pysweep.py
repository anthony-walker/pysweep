
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

    max_swept_steps = query_gpu_z()
    print(max_swept_steps)
    #Creating major axis to split for ranks
    # plane_shape = np.shape(y0)
    # new_shape = (2*max_swept_step,)+init_shape
    # rank_axis = plane_shape.index(max(plane_shape))
    # part_shape = list(plane_shape)
    # part_shape[rank_axis] = int(plane_shape[rank_axis]/num_ranks)
    # part_shape = (2*max_swept_step,)+part_shape
    # rank_axis+=1
    #
    # if rank == master_rank:
    #     #Creating working array
    #     comp_grid = np.zeros(new_shape,dtype=np.float32)  #creating new computational working array
    #     y0 = y0.astype(np.float32)
    #     comp_grid[0] = y0   #Setting computational grid start to initial conditions
    #     comp_grid = np.split(comp_grid,num_ranks,axis=rank_axis)
    #     #Splitting data among ranks
    # else:
    #     comp_grid = None
    #
    # rank_data = comm.scatter(rank_data,root=master_rank)
    # print(rank,": ",np.shape(rank_data))
    #Initial setup

    if rank == master_rank:

        #Creating work element size with max swept step
        # swept_steps = tuple()   #creating temporary tuple()
        # for b_size in block_size:
        #     swept_steps += (int(b_size/(2*ops)),)
        # max_swept_step = min(swept_steps)  #max allowable swept step
        # block_size = (max_swept_step,)+block_size   #updating blocksize
        # del swept_steps         #Deleting swept steps not needed
        #
        # #Creating working array
        # init_shape = np.shape(y0)    #Shape of initial conditions
        # new_shape = (2*max_swept_step,)+init_shape
        # comp_grid = np.zeros(new_shape,dtype=np.float32)  #creating new computational working array
        # y0 = y0.astype(np.float32)
        # comp_grid[0] = y0   #Setting computational grid start to initial conditions
        # # print("Computational Grid Shape: ",newShape)
        #
        #
        # device_ids = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
        # arch_2D_split(0,comp_grid,block_size)
        #Splitting working array by time
        # work_end = deque()
        # blocks = deque()
        # workers = deque()
        # working_block = np.empty(block_size)
        #Getting device attributes
        GPUtil.showUtilization()
        dev_attrs = dev.getDeviceAttrs(0)
        # for key in dev_attrs:
        #     print(key,dev_attrs[key])
    else:
        pass
# grid_shape = np.shape(grid)
#
# gpu_mpc_count = curr_dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)


def query_gpu_z():
    device_ids = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
    max_swept_step = 1e6
    for dev_id in device_ids:
        curr_dev = cuda.Device(dev_id)
        curr_step = curr_dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z)
        if max_swept_step > curr_step:
            max_swept_step = curr_step
    return max_swept_step

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


if __name__ == "__main__":
    dims = (4*256,6*256,4)
    y0 = np.zeros(dims)+1
    dy = [0.1,0.1]
    t0 = 0
    t_b = 1
    dt = 0.001
    order = 2
    block_size = (256,256)
    sweep(y0,dy,t0,t_b,dt,block_size,2,0,40)

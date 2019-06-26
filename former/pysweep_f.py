
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#DataStructure and Math imports
import math
import numpy as np
import space_solvers as ss

#Parallel programming imports
import pycuda as cuda
from mpi4py import MPI
from collections import deque

#My Imports
import dev

class srk2(object):
    """Use this class to create a swept rule integrator - 2nd Order Runge-Kutta."""
    #-----------------------------Constructor Methods---------------------------
    @classmethod
    def pde(cls, dfdt, t0, y0, t_b, d, dt, bc):
        return cls(dfdt, t0, y0, t_b, d, dt, bc)

    def __init__(self, dfdt, t0, y0, t_b, d, dt, bc):
        """This function creates variables from the given class method data."""
        comm = MPI.COMM_WORLD
        masterRank = 0
        numRanks = comm.Get_size()
        rank = comm.Get_rank()
        if rank == masterRank:
            self.dfdt = dfdt    #This is the function to be solved
            self.t0 = t0        #This is the initial time
            self.y0 = y0        #This is the array of initial values
            self.t_b = t_b      #This is the bounding time.
            self.d = d          #This is an array of dx, dy, dz (as needed)
            self.bc = bc        #This is an array of boundary condition indicators
            self.grid = self.create_comp_grid(y0,t0,dt,t_b)  #creating grid in space and time
            self.cgs = np.array_split(self.grid,numRanks)   #Spliting grids with numpy
            #As long as cgs order is maintained, it can be rebuilt with concatenate
            self.solution = np.concatenate(self.cgs)
        dev_attrs = dev.getDeviceAttrs(0)
        print(rank,": ",dev_attrs["DEVICE_NAME"])
        self.swp_t = 0   #Initial swept time


    #-----------------------------CUDA Functions-------------------------------
    def cuda_step(self):
        """Use this function to step with cuda"""
        pass

    def cuda_fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def cuda_fill_max(self):
        """Use this to fill in data up to the given time"""
        pass

    #-----------------------------MPI Functions--------------------------------
    def mpi_step(self):
        """Use this function to step with mpi."""
        self.dfdt

    def mpi_fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def mpi_fill_max(self):
        """Use this to fill in data up to the given time"""
        pass

    #-----------------------------Non-Parallel---------------------------------
    def step():
        """Use this function to solve with without sweep and not in parallel"""
        pass

    def fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def fill_max(self):
        """Use this to fill in data up to the given time"""
        pass


    #-------------------------------Domain functions---------------------------
    def create_comp_grid(self,y0,t0,dt,t_b):
        """Use this function to create a computational grid for the swept rule."""
        initShape = np.shape(y0)    #Shape of initial conditions
        newShape = (int((t_b-t0)/dt),)+initShape
        cGrid = np.zeros(newShape)
        cGrid[0] = y0
        return cGrid

    def bcIdxWrap():
        """Use this to wrap the indices."""

    #--------------------------------General functions-------------------------
    def complete():
        """Use this function to complete the process and return the data."""
        #This function should include clean up and return of the data
        pass


if __name__ ==  "__main__":

    dims = (10,10,6)
    y0 = np.zeros(dims)+1
    t0 = 0
    t_b = 0.02
    # pde(cls, dfdt, t0, y0, t_b, d, dt, bc):
    srk2.pde(ss.ex_dfdt,t0,y0,t_b,(0.1,0.1),0.01,[0,0,0,0])

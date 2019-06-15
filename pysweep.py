
#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#DataStructure and Math imports
import math
import numpy as np
import space_solvers as ss
from grid_creation import *

#Parallel programming imports
import pycuda as pc
from mpi4py import MPI

class srk2(object):
    """Use this class to create a swept rule integrator - 2nd Order Runge-Kutta."""
    #-----------------------------Constructor Methods---------------------------
    @classmethod
    def pde(cls, dfdt, t0, y0, t_b, d, dt, bc):
        return cls(dfdt, t0, y0, t_b, d, dt, bc,ss)

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

    #--------------------------------General functions-------------------------
    def return():
        """Use this function to complete the process and return the data."""
        #This function should include clean up and return of the data
        pass


if __name__ ==  "__main__":

    dims = (100,100,6)
    y0 = np.zeros(dims)
    t0 = 0
    t_b = 0.4
    srk2.pde(ss.ex_dfdt,t0,y0,t_b,0,0,0,5)

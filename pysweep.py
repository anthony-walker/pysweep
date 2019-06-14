#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#DataStructure and Math imports
import math
import numpy as np
import space_solvers as ss
#Parallel programming imports
import pycuda as pc
from mpi4py import MPI

class srk2(object):
    """Use this class to create a swept rule integrator - 2nd Order Runge-Kutta."""

    #Constructor Methods
    @classmethod
    def pde(cls, dfdt, t0, y0, t_b, d, dt, bc, ss):
        "Initialize MyData from a file"
        return cls(dfdt, t0, y0, t_b, d, dt, bc,ss)

    @classmethod
    def ode(cls, dfdt, t0, y0, t_b, dt, ss):
         "Initialize MyData from a dict's items"
         d,bc,ss = None
         return cls(dfdt, t0, y0, t_b, d, dt, bc, ss)

    def __init__(self, dfdt, t0, y0, t_b, d, dt, bc, ss):
        """This function creates variables from the given class method data."""
        comm = MPI.COMM_WORLD
        masterRank = 0
        numRanks = comm.Get_size()
        rank = comm.Get_rank()
        if rank == masterRank:
            self.dfdt = dfdt
            self.t0 = t0
            self.y0 = y0
            self.t_b = t_b
            self.d = d
            self.bc = bc

    #CUDA Functions
    def cuda_step(self):
        """Use this function to step with cuda"""
        pass

    def cuda_fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def cuda_fill_max(self):
        """Use this to fill in data up to the given time"""
        pass


    #MPI Functions
    def mpi_step(self):
        """Use this function to step with mpi."""
        pass


    def mpi_fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def mpi_fill_max(self):
        """Use this to fill in data up to the given time"""
        pass



    #Non-Parallel
    def step():
        """Use this function to solve with without sweep and not in parallel"""
        pass

    def fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def fill_max(self):
        """Use this to fill in data up to the given time"""
        pass





if __name__ ==  "__main__":

    dims = (100,100,6)
    y0 = np.zeros(dims)
    t0 = 0
    t_b = 0.4
    srk2.pde(ss.ex_dfdt,t0,y0,t_b,0,0,0,5)

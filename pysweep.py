#Programmer: Anthony Walker

#PySweep is a package used to implement the swept rule for solving PDEs

#DataStructure and Math imports
import math
import numpy as np

#Parallel programming imports
import pycuda as pc
import mpi4py as MPI



class srk2(object):
    """Use this class to create a swept rule integrator - 2nd Order Runge-Kutta."""

    #Constructor Methods
    @classmethod
    def pde(cls, sfun, t0, y0, t_b, d, dt, bc):
        "Initialize MyData from a file"
        return cls(sfun, t0, y0, t_b, d, dt, bc)

    @classmethod
    def ode(cls, sfun, t0, y0, t_b, dt):
         "Initialize MyData from a dict's items"
         d,bc = None
         return cls(sfun, t0, y0, t_b, d, dt, bc)

    def __init__(self, sfun, t0, y0, t_b, d, dt, bc):
        """This function creates variables from the given class method data."""
        self.sfun = sfun
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
    def np_step():
        """Use this function to solve with without sweep and not in parallel"""
        pass

    def np_fill_level(self):
        """Use this to fill in data up to the given time"""
        pass

    def np_fill_max(self):
        """Use this to fill in data up to the given time"""
        pass





if __name__ ==  "__main__":

    def ex_fcn(arr,d,shape):
        """This is an example function for implementation."""
        ddt = np.zeros(arr.shape())
        for x in shape[0]:
            for y in shape[1]:
                ddt[x,y] = (arr[x+1]+arr[x])/d[0]+(arr[y+1]+arr[y])/d[1]
        return ddt

    dims = (100,100,6)
    y0 = np.zeros(dims)
    t0 = 0
    t_b = 0.4
    srk2.pde(ex_fcn,t0,y0,t_b,0,0,0)

#Programmer: Anthony Walker
#README: This file deals with creation and decomposition of the computational grid
#The dimensions of the computational grid are 2 times the number of
#spatial dimensions. This is because PDEs also change in time and each location
#multiple values such as temperature pressure and etc...

#imports
import numpy as np
from collections import deque

def create_comp_grid(y0,t0,dt,t_b):
    """Use this function to create a computational grid for the swept rule."""
    initShape = np.shape(y0)    #Shape of initial conditions
    newShape = (int((t_b-t0)/dt),)+initShape
    cGrid = np.zeros(newShape)
    cGrid[0] = y0
    cGS = cg_simple_decomp(cGrid)

    return cGS

def cg_simple_decomp(cGrid):
    """Use this function to decompose the working array for swept analysis."""
    cGS = deque()

    return cGS

def stage1_csd():
    """This is the first stage to domain of dependence decompositions."""


def cg_simple_rebuild(cGrids):
    """Use this function to rebuild the working arrays for swept analysis."""
    pass

if __name__ == "__main__":
    y0 = np.zeros((10,10,7))+1
    t0 = 0
    t_b = 1
    dt = 0.1
    create_comp_grid(y0,t0,dt,t_b)

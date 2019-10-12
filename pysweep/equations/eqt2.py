#Programmer: Anthony Walker
#This file contains a test step function for debugging the swept rule

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts+1,vSlice,idx,idy)  #next step index
        cidx = (ts,vSlice,idx,idy)
        pidx = (ts-1,vSlice,idx,idy) #next step index
        if (gts+1)%2==0: #Corrector
            state[ntidx] = (state[pidx])
        else:
            state[ntidx] = (state[cidx])
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    pass

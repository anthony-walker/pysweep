#Programmer: Anthony Walker
#This file contains a test step function for debugging the swept rule

import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass

def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    TSO = 2
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts,vSlice,idx,idy)  #next step index
        cidx = (ts-1,vSlice,idx,idy)
        eidx = (ts-1,vSlice,idx+1,idy)
        widx = (ts-1,vSlice,idx-1,idy)
        nidx = (ts-1,vSlice,idx,idy+1)
        sidx = (ts-1,vSlice,idx,idy-1)
        pidx = (ts-TSO,vSlice,idx,idy) #next step index
        if (gts)%TSO==0: #Corrector
            state[ntidx] = (state[pidx])+2
        else: #Predictor
            state[ntidx] = (state[nidx]+state[sidx]+state[eidx]+state[widx])/4+2
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    pass

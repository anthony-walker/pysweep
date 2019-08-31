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
    for idx in iidx:
        nidx = (ts+1,vSlice)+idx  #next step index
        cidx = (ts,vSlice)+idx
        eidx = (ts,vSlice,idx[0]+1,idx[1])
        widx = (ts,vSlice,idx[0]-1,idx[1])
        nidx = (ts,vSlice,idx[0],idx[1]+1)
        sidx = (ts,vSlice,idx[0],idx[1]-1)
        pidx = (ts-1,vSlice)+idx  #next step index
        # state[nidx] = state[cidx]+state[ss+(idx[0],)+((idx[1]+1),)]+state[ss+(idx[0],)+((idx[1]-1),)]
        # state[nidx] += state[ss+((idx[0]+1),)+(idx[1],)]+state[ss+((idx[0]-1),)+(idx[1],)]
        if gts%2!=0:
            state[nidx] = (state[nidx]+state[sidx]+state[eidx]+state[widx])/4
        else:
            state[nidx] = state[cidx]+1
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy,gam = args
    if gpu:
        keys = "DT","DX","DY","GAMMA","GAM_M1"
        nargs = args[2:]+(gam-1,)
        fc = lambda x:np.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        global dtdx
        dtdx = dt/dx
        global dtdy
        dtdy = dt/dy
        global gamma
        gamma = gam

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
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts+1,vSlice,idx,idy)  #next step index
        cidx = (ts,vSlice,idx,idy)
        eidx = (ts,vSlice,idx+1,idy)
        widx = (ts,vSlice,idx-1,idy)
        nidx = (ts,vSlice,idx,idy+1)
        sidx = (ts,vSlice,idx,idy-1)
        eeidx = (ts,vSlice,idx+2,idy)
        wwidx = (ts,vSlice,idx-2,idy)
        nnidx = (ts,vSlice,idx,idy+2)
        ssidx = (ts,vSlice,idx,idy-2)
        pidx = (ts-1,vSlice,idx,idy) #next step index
        if (gts+1)%2==0: #Corrector
            state[ntidx] = (state[pidx])+2
        else:
            state[ntidx] = (state[nidx]+state[sidx]+state[eidx]+state[widx]+state[nnidx]+state[ssidx]+state[eeidx]+state[wwidx])/8+1
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

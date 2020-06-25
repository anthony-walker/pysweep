#Programmer: Anthony Walker
#This file contains a test step function for debugging the swept rule

import numpy, h5py
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass

def step(state,iidx,ts,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    half = 0.5
    TSO = 2
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts+1,vSlice,idx,idy)  #next step index
        cidx = (ts,vSlice,idx,idy)
        pidx = (ts-1,vSlice,idx,idy) #next step index
        state[ntidx] = state[cidx]+1
        # if (globalTimeStep+1)%TSO==0: #Corrector
        #     state[ntidx] = (state[pidx])+1
        # else: #Predictor
        #     state[ntidx] = state[cidx]+1
    return state

def createInitialConditions(nv,nx,ny,filename="exampleConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file."""
    with h5py.File(filename,"w") as f:
        f.create_dataset("data",(nv,nx,ny),data=numpy.ones((nv,nx,ny)))
    return filename

def set_globals(gpu,*args,source_mod=None):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy = args
    if gpu:
        keys = "DT","DX","DY"
        nargs = args[2:]
        fc = lambda x:numpy.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        global dtdx
        dtdx = dt/dx
        global dtdy
        dtdy = dt/dy

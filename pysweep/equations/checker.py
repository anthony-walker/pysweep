#Programmer: Anthony Walker
#This file contains a test step function for debugging the swept rule

import numpy, h5py, mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    if scheme:
        return checkerOneStep(state,iidx,arrayTimeIndex,globalTimeStep)
    else:
        return checkerTwoStep(state,iidx,arrayTimeIndex,globalTimeStep)

def checkerOneStep(state,iidx,arrayTimeIndex,globalTimeStep):
    """Use this function as the one step checker pattern"""
    vs = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (arrayTimeIndex+1,vs,idx,idy)  #next step index
        state[ntidx] = state[arrayTimeIndex,vs,idx+1,idy]
        state[ntidx] += state[arrayTimeIndex,vs,idx-1,idy]
        state[ntidx] += state[arrayTimeIndex,vs,idx,idy+1]
        state[ntidx] += state[arrayTimeIndex,vs,idx,idy-1]
        state[ntidx] /= 4
    return state

def checkerTwoStep(state,iidx,arrayTimeIndex,globalTimeStep):
    """Use this function as the two step checker pattern"""
    vs = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (arrayTimeIndex+1,vs,idx,idy)  #next step index
        if (globalTimeStep+1)%2==0:
            state[ntidx] = state[arrayTimeIndex,vs,idx+1,idy]
            state[ntidx] += state[arrayTimeIndex,vs,idx-1,idy]
            state[ntidx] += state[arrayTimeIndex,vs,idx,idy+1]
            state[ntidx] += state[arrayTimeIndex,vs,idx,idy-1]
            state[ntidx] /= 4
        else:
            state[ntidx] = state[arrayTimeIndex,vs,idx,idy]
    return state

def createInitialConditions(nv,nx,ny,filename="checkerConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file."""
    comm = MPI.COMM_WORLD
    data = numpy.zeros((nv,nx,ny))
    for i in range(0,nx,2):
        for j in range(0,ny,2):
            data[:,i,j]=1
    for i in range(1,nx,2):
        for j in range(1,ny,2):
            data[:,i,j]=1
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",data.shape,data=data)
    return filename

def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,scheme #true for one step
    t0,tf,dt,dx,dy,scheme = args
    if source_mod is not None:
        keys = "DT","DX","DY"
        nargs = args[2:]
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,bytes(scheme))

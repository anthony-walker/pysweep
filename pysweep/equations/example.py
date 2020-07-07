#Programmer: Anthony Walker
#This file contains a test step function for debugging the swept rule

import numpy, h5py, mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass

scheme = True

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    if scheme: #pseude FE
        for idx,idy in iidx:
            state[arrayTimeIndex+1,:,idx,idy] = state[arrayTimeIndex,:,idx,idy]+1
    else: #pseudo RK2
        addition,timeChange = (1,0) if globalTimeStep%2==0 else  (2,1)
        for idx,idy in iidx:
            state[arrayTimeIndex+1,:,idx,idy] = state[arrayTimeIndex-timeChange,:,idx,idy]+addition


def createInitialConditions(nv,nx,ny,filename="exampleConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file."""
    comm = MPI.COMM_WORLD
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",(nv,nx,ny),data=numpy.zeros((nv,nx,ny)))
    return filename

def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,scheme #true for one step
    t0,tf,dt,dx,dy,scheme = args
    if source_mod is not None:
        keys = "DT","DX","DY"
        nargs = args[2:]
        fc = lambda x:numpy.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,numpy.intc(scheme))
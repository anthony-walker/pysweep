
import numpy, h5py, os, sys
import mpi4py.MPI as MPI
import pycuda.driver as cuda
import pysweep.core.io as io
scheme = True
pi = numpy.pi
piSq = pi*pi

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """Use this function to solve the HDE with RK2."""
    coeff,timechange =  (1,1) if globalTimeStep%2==0 else (0.5,0)  #True - Final Step, False- Intermediate Step
    for idx,idy in iidx:
        state[arrayTimeIndex+1,0,idx,idy] = coeff*centralDifference(state[arrayTimeIndex,0],idx,idy)+state[arrayTimeIndex-timechange,0,idx,idy]
    
def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy #true for FE
    t0,tf,dt,dx,dy = args
    if source_mod is not None:
        keys = "DT","DX","DY"
        nargs = args[2:]
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))

def writeOut(arr):
        for row in arr:
            sys.stdout.write("[")
            for item in row:
                if item==0:
                    sys.stdout.write("\033[1;36m")
                else:
                    sys.stdout.write("\033[1;31m")
                sys.stdout.write("%.1e, "%item)
            sys.stdout.write("]\n")
        sys.stdout.write("\n")


def centralDifference(state,idx,idy):
    """Use this function to solve the HDE with a 3 point central difference."""
    nx,ny = state.shape
    secondDerivativeX = (state[idx+1,idy]-2*state[idx,idy]+state[idx-1,idy])/100
    secondDerivativeY = (state[idx,idy+1]-2*state[idx,idy]+state[idx,idy-1])/100
    return secondDerivativeX+secondDerivativeY

def createInitialConditions(npx,npy,t=0,filename="halfConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    data = numpy.zeros((1,npx,npy))
    for i,ids in enumerate(numpy.ndindex(data.shape[1:])):
        idx,idy = ids
        data[0,idx,idy] = i/100
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",data.shape,data=data)
    return filename

if __name__ == "__main__":
    pass
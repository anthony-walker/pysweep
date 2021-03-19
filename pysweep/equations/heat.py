import numpy, h5py, os, sys
import itertools
import mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
except Exception as e:
    pass
#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib import animation, rc

#globals
scheme = True
pi = numpy.pi
piSq = pi*pi

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - array time index for state
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    if scheme:
        forwardEuler(state,iidx,arrayTimeIndex,globalTimeStep)
    else:
        return rungeKuttaTwo(state,iidx,arrayTimeIndex,globalTimeStep)

def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,alpha,scheme,courant #true for FE
    t0,tf,dt,dx,dy,alpha,scheme = args
    courant = alpha*dt/dx/dx #technically not the courant number
    if source_mod is not None:
        keys = "DT","DX","DY","ALPHA"
        nargs = args[2:]
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,numpy.intc(scheme))
    
def forwardEuler(state,iidx,arrayTimeIndex,globalTimeStep):
    """Use this function to solver the HDE with forward Euler."""
    for idx,idy in iidx:
        state[arrayTimeIndex+1,0,idx,idy] = centralDifference(state[arrayTimeIndex,0],idx,idy)+state[arrayTimeIndex,0,idx,idy]

def rungeKuttaTwo(state,iidx,arrayTimeIndex,globalTimeStep):
    """Use this function to solve the HDE with RK2."""
    coeff,timechange =  (1,1) if globalTimeStep%2==0 else (0.5,0)  #True - Final Step, False- Intermediate Step
    for idx,idy in iidx:
        state[arrayTimeIndex+1,0,idx,idy] = coeff*centralDifference(state[arrayTimeIndex,0],idx,idy)+state[arrayTimeIndex-timechange,0,idx,idy]
    
def centralDifference(state,idx,idy):
    """Use this function to solve the HDE with a 3 point central difference."""
    nx,ny = state.shape
    secondDerivativeX = courant*(state[(idx+1)%nx,idy]-2*state[idx,idy]+state[idx-1,idy])
    secondDerivativeY = courant*(state[idx,(idy+1)%ny]-2*state[idx,idy]+state[idx,idy-1])
    return secondDerivativeX+secondDerivativeY

def createInitialConditions(npx,npy,alpha=0.1,t=0,filename="heatConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    csize = comm.Get_size()
    X = numpy.linspace(0,1,csize+1,endpoint=True)[rank:rank+2]
    Idx = numpy.array_split([i for i in range(npx)],csize)[rank]
    #Get point specific x and y
    X = numpy.linspace(X[0],X[1],len(Idx),endpoint=True)
    Y = numpy.linspace(0,1,npy,endpoint=True)
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        initialConditions = hf.create_dataset("data",(1,npx,npy))
        for i,x in enumerate(X):
            gi = Idx[i]
            for j,y in enumerate(Y):
                initialConditions[0,gi,j] = analyticalEquation(x,y,t=t,alpha=alpha)
    comm.Barrier()
    return filename

def analyticalEquation(x,y,t,alpha=0.1):
    """Use this to fill hdf5 IC."""
    m = n = 2
    return numpy.sin(m*pi*x)*numpy.sin(n*pi*y)*numpy.exp(-(m*m+n*n)*pi*pi*alpha*t)

def analytical(npx,npy,t,alpha=0.1):
    """Use this function to generate the prescribed analytical solution analytical solution.
    x: 0->1
    y: 0->1
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of interest
    """
    X = numpy.linspace(0,1,npx,endpoint=False)
    Y = numpy.linspace(0,1,npy,endpoint=False)
    uShape = (1,numpy.shape(X)[0],numpy.shape(Y)[0])
    u = numpy.zeros(uShape)
    m = 2 #m and n = 2 makes the function periodic in 0->1
    n = 2
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            u[0,i,j] = numpy.sin(m*pi*x)*numpy.sin(n*pi*y)*numpy.exp(-(m*m+n*n)*pi*pi*alpha*t)
    return u,X,Y

def getFinalTime(npx,tsteps,alpha):
    d = 0.1
    alpha = 0.01
    dx = 1/npx
    dt = float(d*dx**2/alpha)
    return dt*tsteps,dt


def verifyOrder(schemeBool=True,fname=None):
    """Use this function to verify the order of Runge-Kutta Second order."""
    global scheme,dt,dx,dy,alpha,courant
    d = 0.1
    courant = d
    scheme = schemeBool
    sizes = [int(i*12) for i in range(1,6)]
    error = []
    dtCases = []
    tf = 0.1
    for arraysize in sizes:
        alpha = 1
        dx = dy = 1/arraysize
        dt = float(d*dx*dx/alpha)
        timesteps = int(tf//dt)
        print(timesteps)
        dtCases.append(dt)
        T,x,y = analytical(arraysize,arraysize,t=0,alpha=alpha)
        numerical = numpy.zeros((2,)+T.shape) if scheme else numpy.zeros((3,)+T.shape)
        numerical[0,:,:,:] = T[0,:,:]
        iidx = list(numpy.ndindex((arraysize,arraysize)))
        for i in range(timesteps): #current time step
            for j in range(1,numerical.shape[0]): #current scheme step
                step(numerical,iidx,j-1,j)
            numerical[0,:,:,:] = numerical[-1,:,:,:]
        T,x,y = analytical(arraysize,arraysize,dt*timesteps,alpha=alpha)
        errArr = numpy.absolute(T[0,:,:]-numerical[-1,0,:,:])
        error.append(numpy.amax(errArr))   
    plt.loglog(dtCases,error,marker='o')
    plt.ylim(10e-10,10e-2)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

if __name__ == "__main__":
    # verifyOrder(True,"forwardEulerHeatOrder.pdf")
    verifyOrder(False)
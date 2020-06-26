
import numpy, h5py, mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
except Exception as e:
    print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
    
scheme = True

def step(state,iidx,ts,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    if scheme:
        return forwardEuler(state,iidx,ts,globalTimeStep)
    else:
        return rungeKutta2(state,iidx,ts,globalTimeStep)


def set_globals(gpu,*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,alpha,scheme
    t0,tf,dt,dx,dy,alpha,scheme = args
    if gpu:
        keys = "DT","DX","DY","ALPHA"
        nargs = args[2:]
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,bytes(scheme))
    

def forwardEuler(state,iidx,ts,globalTimeStep):
    """Use this function to solver the HDE with forward Euler."""
    for idx,idy in iidx:
        d2dx,d2dy = centralDifference(state,ts,idx,idy)
        state[ts+1,0,idx,idy] = alpha*dt*(d2dx+d2dy) + state[ts,0,idx,idy]
    return state

def rungeKutta2(state,iidx,ts,globalTimeStep):
    """Use this function to solve the HDE with RK2."""
    pass


def centralDifference(state,ts,idx,idy):
    """Use this function to solve the HDE with a 3 point central difference."""
    secondDerivativeX = (state[ts,0,idx+1,idy]-2*state[ts,0,idx,idy]+state[ts,0,idx-1,idy])/(dx*dx)
    secondDerivativeY = (state[ts,0,idx,idy+1]-2*state[ts,0,idx,idy]+state[ts,0,idx,idy-1])/(dy*dy)
    return secondDerivativeX,secondDerivativeY

def createInitialConditions(npx,npy,t=0,filename="heatConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    u = analytical(npx,npy,t)
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",u.shape,data=u)
    return filename

def analytical(npx,npy,t):
    """Use this function to generate the prescribed analytical solution analytical solution.
    x: 0->1
    y: 0->1
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of interest
    """
    X = numpy.linspace(0,1,npx)
    Y = numpy.linspace(0,1,npy)
    uShape = (1,numpy.shape(X)[0],numpy.shape(Y)[0])
    u = numpy.zeros(uShape)
    pi = numpy.pi
    m = 2 #m and n = 2 makes the function periodic in 0->1
    n = 2
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            u[0,i,j] = numpy.sin(m*pi*x)*numpy.sin(n*pi*y)*numpy.exp(-(m^2+n^2)*t)
    return u

if __name__ == "__main___":
    uIC = analytical(0.1,0.1,1)

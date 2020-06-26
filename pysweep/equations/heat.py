
import numpy, h5py, mpi4py.MPI as MPI

def step(state,iidx,ts,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    statements = {0:statement0,1:statement1}
    # return statements[globalTimeStep%2](state,iidx,ts)
    return statementf(state,iidx,ts)

def createInitialConditions(dx,dy,t=0,filename="heatConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    dx: step size in x direction
    dy: step size in y direction
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    u = analytical(dx,dy,t)
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",u.shape,data=u)
    return filename

def analytical(dx,dy,t):
    """Use this function to generate the prescribed analytical solution analytical solution.
    x: 0->1
    y: 0->1
    args:
    dx: spatial discretization in horizontal direction.
    dy: spatial discretization in vertical direction.
    t: time of interest
    """
    X = numpy.arange(0,1,dx)
    Y = numpy.arange(0,1,dy)
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
    analytical(0.1,0.1,1)
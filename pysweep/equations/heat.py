
import numpy, h5py 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib import animation, rc
# import mpi4py.MPI as MPI
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
    global dt,dx,dy,alpha,scheme #true for FE
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
    nt,nv,nx,ny = state.shape
    secondDerivativeX = (state[ts,0,(idx+1)%nx,idy]-2*state[ts,0,idx,idy]+state[ts,0,(idx-1)%nx,idy])/(dx*dx)
    secondDerivativeY = (state[ts,0,idx,(idy+1)%ny]-2*state[ts,0,idx,idy]+state[ts,0,idx,(idy-1)%ny])/(dy*dy)
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
    X = numpy.linspace(0,1,npx,endpoint=False)
    Y = numpy.linspace(0,1,npy,endpoint=False)
    uShape = (1,numpy.shape(X)[0],numpy.shape(Y)[0])
    u = numpy.zeros(uShape)
    pi = numpy.pi
    m = 2 #m and n = 2 makes the function periodic in 0->1
    n = 2
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            u[0,i,j] = numpy.sin(m*pi*x)*numpy.sin(n*pi*y)*numpy.exp(-(m^2+n^2)*t)
    return u,X,Y

class HeatGIF(object):
    
    def __init__(self):
        super(HeatGIF,self).__init__()
        self.cT = []

    def appendTemp(self,T):
        self.cT.append(T)

    def setXY(self,X,Y):
        """Use this function to set X and Y."""
        self.X = X
        self.Y = Y
    
    def makeGif(self):
        fig =  plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=45, azim=25)
        ax.set_ylim(numpy.amin(self.Y), numpy.amax(self.Y))
        ax.set_ylim(numpy.amin(self.Y), numpy.amax(self.Y))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
        fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=numpy.linspace(-1,1,10))
        animate = lambda i: ax.plot_surface(self.X,self.Y,self.cT[i],cmap=cm.inferno)
        frames = len(self.cT)
        anim = animation.FuncAnimation(fig,animate,frames)
        anim.save("heat.gif",writer="imagemagick")

if __name__ == "__main__":
    alpha = 1 #0.00016563 #Of pure silver
    dt = 0.1
    npx = 11
    npy = 11
    t0 = 0
    tf = 10
    uIC,X,Y = analytical(npx,npy,t0)
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    globs = t0,tf,dt,
    set_globals(False,t0,tf,dt,dx,dy,alpha,True)
    U = numpy.zeros((2,)+uIC.shape)
    U[0,:,:,:] = uIC[:,:,:]
    idxs = numpy.ndindex((npx,npy))
    X, Y = numpy.meshgrid(X, Y)
    hgf = HeatGIF()
    hgf.setXY(X,Y)
    for i,time in enumerate(numpy.arange(t0,tf,dt)):
        U = step(U,idxs,0,i)
        uA,_,__ = analytical(npx,npy,time)
        # print(U[1]-U[0])
        U[0] = U[1]
        hgf.appendTemp(uA[0])
    hgf.makeGif()

import numpy, h5py, os, sys
import mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
except Exception as e:
    print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
    
scheme = True
pi = numpy.pi
piSq = pi*pi

def step(state,iidx,ts,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    if scheme:
        state = forwardEuler(state,iidx,ts,globalTimeStep)
        return state
    else:
        return rungeKutta2(state,iidx,ts,globalTimeStep)


def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,alpha,scheme #true for FE
    t0,tf,dt,dx,dy,alpha,scheme = args
    if source_mod is not None:
        keys = "DT","DX","DY","ALPHA"
        nargs = args[2:]
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,numpy.intc(scheme))
    

def forwardEuler(state,iidx,ts,globalTimeStep):
    """Use this function to solver the HDE with forward Euler."""
    d2dx = numpy.zeros(state.shape[2:])
    d2dy = numpy.zeros(state.shape[2:])
    source = numpy.zeros(state.shape[2:])
    for idx,idy in iidx:
        d2dx[idx,idy],d2dy[idx,idy] = centralDifference(state[0,0],idx,idy)
    state[ts+1,0,:,:] = d2dx[:,:]+d2dy[:,:]+state[ts,0,:,:]
    return state

def writeOut(arr):
        for row in arr:
            sys.stdout.write("[")
            for item in row:
                sys.stdout.write("%.3f, "%item)
            sys.stdout.write("]\n")
        sys.stdout.write("\n")


def rungeKutta2(state,iidx,ts,globalTimeStep):
    """Use this function to solve the HDE with RK2."""
    pass

def centralDifference(state,idx,idy):
    """Use this function to solve the HDE with a 3 point central difference."""
    nx,ny = state.shape
    secondDerivativeX = alpha*dt*(state[(idx+1)%nx,idy]-2*state[idx,idy]+state[(idx-1)%nx,idy])/(dx*dx)
    secondDerivativeY = alpha*dt*(state[idx,(idy+1)%ny]-2*state[idx,idy]+state[idx,(idy-1)%ny])/(dy*dy)
    return secondDerivativeX,secondDerivativeY

def createInitialConditions(npx,npy,alpha=0.1,t=0,filename="heatConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    u,X,Y = analytical(npx,npy,t,alpha=alpha)
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        hf.create_dataset("data",u.shape,data=u)
    return filename

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

def sourceTerm(idx,idy,globalTimeStep):
    x = idx*dx
    y = idy*dy
    t = dt*globalTimeStep
    expon = numpy.exp(-8*piSq*alpha*t)
    trig = numpy.sin(2*pi*x)*numpy.sin(2*pi*y)
    sx = -4*alpha*piSq*expon*trig
    sy = -4*alpha*piSq*expon*trig
    st = -8*piSq*alpha*expon*trig
    sTotal = dt*(sx+sy+st)
    return sTotal

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
    
    def animate(self,i):
        ax.cla() #clear off axis
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.plot_surface(self.X,self.Y,self.cT[i],cmap=cm.inferno)

    def makeGif(self,name="heat.gif"):
        global ax
        fig =  plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=45, azim=25)
        
        # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
        fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=numpy.linspace(-1,1,10))
        
        frames = len(self.cT)
        anim = animation.FuncAnimation(fig,self.animate,frames)
        anim.save(name,writer="imagemagick")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits import mplot3d
    from matplotlib import animation, rc
    os.system("rm heat.gif") #Remove old heat gif
    alpha = 0.1#0.00016563 #Of pure silver
    dt = 0.0001
    npx = 11
    npy = 11
    t0 = 0
    tf = 0.5
    uIC,X,Y = analytical(npx,npy,t0)
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    globs = t0,tf,dt,
    set_globals(False,t0,tf,dt,dx,dy,alpha,True)
    U = numpy.zeros((2,)+uIC.shape)
    U[0,:,:,:] = uIC[:,:,:]
    idxs = list(numpy.ndindex((npx,npy)))
    X, Y = numpy.meshgrid(X, Y)
    hgf = HeatGIF()
    hgf.setXY(X,Y)
    hgf.appendTemp(U[0,0])
    error = [numpy.amax(U[0]-uIC),]
    for i,time in enumerate(numpy.arange(t0,tf+dt,dt)):
        U = step(U,idxs,0,i)
        uA,_,__ = analytical(npx,npy,time+dt)
        # print(numpy.amax(U[0]-U[1]))
        U[0,:,:,:] = numpy.copy(U[1,:,:,:])
        U[1,:,:,:] = 0
        error.append(numpy.amax(U[0]-uA))
        hgf.appendTemp(U[0,0])
    print(numpy.amax(error))
    # hgf.makeGif()
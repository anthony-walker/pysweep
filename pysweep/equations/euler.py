"""
Programmer: Anthony Walker
Use this file to solve the Euler's equations with 
fixed stencil conservative finite difference approach (Central 4th order)
"""

import sys,itertools,numpy,h5py,mpi4py.MPI as MPI,pycuda.driver as cuda
#----------------------------------Globals-------------------------------------#
gamma = 1.4
dtdx = 0
dtdy = 0
gM1 = 0.4
#----------------------------------End Globals-------------------------------------#

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    ops = 2
    coeff,timechange =  (1,1) if globalTimeStep%2==0 else (0.5,0)  #True - Final Step, False- Intermediate 
    for idx,idy in iidx:
        fluxx = dtdx*(getFluxX(state[arrayTimeIndex],idx,idy)-getFluxX(state[arrayTimeIndex],idx-1,idy))
        fluxy = dtdy*(getFluxY(state[arrayTimeIndex],idx,idy)-getFluxY(state[arrayTimeIndex],idx,idy-1))
        state[arrayTimeIndex+1,:,idx,idy] = state[arrayTimeIndex,:,idx,idy]-fluxx-fluxy
    return state

def getFluxX(state,idx,idy):
    """This function finds the x derivative."""
    dfdx = numpy.zeros(4,)
    dvars = numpy.asarray([getVars(state[:,i,idy],True) for i in [idx-1,idx,idx+1,idx+2]])
    dfdx[:] = -dvars[0,:]/12+7*dvars[1,:]/12+7*dvars[2,:]/12-dvars[3,:]/12
    return dfdx

def getFluxY(state,idx,idy):
    """This function finds the x derivative."""
    dgdy = numpy.zeros(4,)
    dvars = numpy.asarray([getVars(state[:,idx,i],False) for i in [idy-1,idy,idy+1,idy+2]])
    dgdy[:] = -dvars[0,:]/12+7*dvars[1,:]/12+7*dvars[2,:]/12-dvars[3,:]/12
    return dgdy

def getVars(state,ifx):
    """Use this function to get derivative variables."""
    P = gM1*(state[3]-0.5*(state[1]*state[1]+state[2]*state[2])/state[0])
    u = state[1]/state[0]
    v = state[2]/state[0]
    if ifx:
        derivativeVariables = state[1],state[0]*u*u+P,state[0]*u*v,(state[3]+P)*u 
    else:
        derivativeVariables = state[1],state[0]*u*v,state[0]*v*v+P,(state[3]+P)*u 
    return numpy.asarray(derivativeVariables)

def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dtdx,dtdy,gamma,gM1
    t0,tf,dt,dx,dy,gamma = args
    dtdx = dtdy = dt/dx
    gM1 = gamma-1
    if source_mod is not None:
        keys = "DT","DX","DY","GAMMA","GAM_M1","DTDX","DTDY"
        nargs = args[2:]+(gamma-1,dt/dx,dt/dy)
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))

def createInitialConditions(npx,npy,t=0,gamma=1.4,filename="eulerConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    comm = MPI.COMM_WORLD
    with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
        initialConditions = hf.create_dataset("data",(4,npx,npy))
        L = 5 #From shu
        xRange = numpy.linspace(-L,L,npx,endpoint=False)
        yRange = numpy.linspace(-L,L,npy,endpoint=False)
        dx = 1/npx
        dy = 1/npy
        for i,x in enumerate(xRange):
            for j,y in enumerate(yRange):
                initialConditions[:,i,j] = analytical(x,y,t,gamma=gamma)
    return filename

    
def analytical(x,y,t,gamma=1.4):
    """This is the primary method to solve the euler vortex that is centered at the origin with periodic boundary conditions
    properties are obtained from the vics object, cvics.
       The center can be changed with x0 and y0.
       the time step or steps can be changed with times but it must be a tuple.
       X & Y are dimensions of the domain.
       npx, and npy are the number of points the in respective direction

       This solution was obtained from

    """
    PI = numpy.pi
    infinityMach = numpy.sqrt(2/gamma)
    alpha = PI/2 #Angle of attack 90 degrees - straight x
    # alpha = 0 #Angle of attack 0 degrees - straight y
    # alpha = PI/4 #Angle of attack 45 degrees
    infinityRho = 1
    infinityP = 1
    infinityT = 1
    Rv = 1 #Vortex radius
    sigma = 1 #Perturbation constant
    beta = infinityMach*5*numpy.sqrt(2)/4/numpy.pi*numpy.exp(0.5)
    #Getting Freestream velocity
    c = numpy.sqrt(gamma*infinityP/infinityRho)
    V_inf = infinityMach*c
    # assert epsilon >= L/Rv*numpy.exp(-L*L/(2*Rv*Rv*sigma*sigma))
    u0 = V_inf*numpy.cos(alpha)
    v0 = V_inf*numpy.sin(alpha)
    #solving for data
    dx0 = x-u0*t
    dy0 = y-v0*t
    # dx0 -= 10*dx0//5 if dx0//5%2!=0 else 5*dx0//5
    f = -1/(2*sigma*sigma)*((dx0/Rv)**2+(dy0/Rv)**2)
    Omega = beta*numpy.exp(f)
    du = -(dy0)*Omega/Rv
    dv = (dx0)*Omega/Rv
    dT = -(gamma-1)*Omega*Omega/2
    #Variables
    rho = (1+dT)**(1/(gamma-1))
    u = u0+du
    v = v0+dv
    P = 1/gamma*(1+dT)**(gamma/(gamma-1))
    return rho,rho*u,rho*v,P/(gamma-1)+0.5*rho*(u*u+v*v)

def getAnalyticalArray(npx,npy,t):
    """Use this function to get an analytical array for testing."""
    state =numpy.zeros((4,npx,npy))
    L = 5 #From shu
    xRange = numpy.linspace(-L,L,npx,endpoint=False)
    yRange = numpy.linspace(-L,L,npy,endpoint=False)
    for i,x in enumerate(xRange):
        for j,y in enumerate(yRange):
            state[:,i,j] = analytical(x,y,t,gamma=gamma)
    return state

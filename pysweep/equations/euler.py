#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way

import sys,itertools,numpy,h5py,mpi4py.MPI as MPI, pysweep.equations.sodShock as sod
try:
    import pycuda.driver as cuda
except Exception as e:
    pass
#----------------------------------Globals-------------------------------------#
gamma = 1.4
dtdx = 0
dtdy = 0
gM1 = 0.4

#----------------------------------End Globals-------------------------------------#
    

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
    comm.Barrier()
    return filename

    
def analytical(x,y,t,gamma=1.4,alpha=numpy.pi/4):
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
    # alpha = PI/2 #Angle of attack 90 degrees - straight x
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

def getPeriodicShock(npx,t,direc=True):
    """Use this function to test against the sod shock tube in 2D.
    This doesn't solve shock interaction, just orients the data so it can be solved periodically temporarily.
    """
    shock = sod.sodShock(t,npx//2,npx,direc)
    return numpy.concatenate((numpy.flip(shock,axis=1),shock),axis=1)

def getShock(npx,t,direc=True):
    """Use this function to test against the sod shock tube in 2D.
    This doesn't solve shock interaction, just orients the data so it can be solved periodically temporarily.
    """
    return sod.sodShock(t,npx,npx,direc)

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    ops=2 #number of points on each side of a give point
    #Finding pressure
    pressure  = getPressureArray(state[arrayTimeIndex],iidx) #For use with swept
    #Solving fluxes
    if globalTimeStep%2==0: #RK2 step
        fluxx = getFluxInX(state[arrayTimeIndex],pressure,iidx,ops)
        fluxy = getFluxInY(state[arrayTimeIndex],pressure,iidx,ops)
        for idx,idy in iidx:
            state[arrayTimeIndex+1,:,idx,idy] = state[arrayTimeIndex-1,:,idx,idy]+(fluxx[:,idx,idy]*dtdx+fluxy[:,idx,idy]*dtdy)
    else: #intermediate step
        fluxx = getFluxInX(state[arrayTimeIndex],pressure,iidx,ops)
        fluxy = getFluxInY(state[arrayTimeIndex],pressure,iidx,ops)
        for idx,idy in iidx:
            state[arrayTimeIndex+1,:,idx,idy] = state[arrayTimeIndex,:,idx,idy]+0.5*(fluxx[:,idx,idy]*dtdx+fluxy[:,idx,idy]*dtdy)
            
#---------------------------------------------Solving functions

def getXStencil(idx,idy,maxLen):
    return ((idx-2)%maxLen,idy),((idx-1)%maxLen,idy),(idx,idy),((idx+1)%maxLen,idy),((idx+2)%maxLen,idy)

def getFluxInX(state,P,iidx,ops):
    """Use this function to get flux in the X direction."""
    flux = numpy.zeros((state.shape[0],state.shape[1],state.shape[2]))
    xlen = state.shape[1]
    vs = slice(0,4,1)
    #west part of stencil
    for idx,idy in iidx:
        ww,w,c,e,ee = getXStencil(idx,idy,xlen)
        left = fluxLimiter(state,(vs,)+w,(vs,)+c,P[c]-P[w],P[w]-P[ww])
        right = fluxLimiter(state,(vs,)+c,(vs,)+w,P[c]-P[w],P[e]-P[c])
        flux[:,idx,idy] += evaluateFluxInX(left,right)
        flux[:,idx,idy] += evaluateSpectral(left,right
        ,True)

    #east part of stencil
    for idx,idy in iidx:
        ww,w,c,e,ee = getXStencil(idx,idy,xlen)
        left = fluxLimiter(state,(vs,)+c,(vs,)+e,P[e]-P[c],P[c]-P[w])
        right = fluxLimiter(state,(vs,)+e,(vs,)+c,P[e]-P[c],P[ee]-P[e])
        flux[:,idx,idy] -= evaluateFluxInX(left,right)
        flux[:,idx,idy] -= evaluateSpectral(left,right,True)
    return flux*0.5

def evaluateFluxInX(left_state,right_state):
    """Use this method to calculation the flux.
    q (state) is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    #Pressures
    PL = getPressure(left_state)
    PR = getPressure(right_state)
    flux = numpy.zeros(left_state.shape)
    flux[0] = left_state[1]+right_state[1]
    flux[1] = left_state[1]**2/left_state[0]+PL+right_state[1]**2/right_state[0]+PR
    flux[2] = left_state[1]*left_state[2]/left_state[0]+right_state[1]*right_state[2]/right_state[0]
    flux[3] = (left_state[3]+PL)*left_state[1]/left_state[0]+(right_state[3]+PR)*right_state[1]/right_state[0]
    return flux

def getYStencil(idx,idy,maxLen):
    return (idx,(idy-2)%maxLen),(idx,(idy-1)%maxLen),(idx,idy),(idx,(idy+1)%maxLen),(idx,(idy+2)%maxLen)

def getFluxInY(state,P,iidx,ops):
    """Use this function to get flux in the X direction."""
    flux = numpy.zeros((state.shape[0],state.shape[1],state.shape[2]))
    ylen = state.shape[2]
    vs = slice(0,4,1)
    #west part of stencil
    for idx,idy in iidx:
        ss,s,c,n,nn = getYStencil(idx,idy,ylen)
        left = fluxLimiter(state,(vs,)+s,(vs,)+c,P[c]-P[s],P[s]-P[ss])
        right = fluxLimiter(state,(vs,)+c,(vs,)+s,P[c]-P[s],P[n]-P[c])
        flux[:,idx,idy] += evaluateFluxInY(left,right)
        flux[:,idx,idy] += evaluateSpectral(left,right,False)

    #east part of stencil
    for idx,idy in iidx:
        ss,s,c,n,nn = getYStencil(idx,idy,ylen)
        left = fluxLimiter(state,(vs,)+c,(vs,)+n,P[n]-P[c],P[c]-P[s])
        right = fluxLimiter(state,(vs,)+n,(vs,)+c,P[n]-P[c],P[nn]-P[n])
        flux[:,idx,idy] -= evaluateFluxInY(left,right)
        flux[:,idx,idy] -= evaluateSpectral(left,right,False)
        
    return flux*0.5

def evaluateFluxInY(left_state,right_state):
    """Use this method to calculation the flux.
    q (state) is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    #Pressures
    PL = getPressure(left_state)
    PR = getPressure(right_state)
    flux = numpy.zeros(left_state.shape)
    flux[0] = left_state[2]+right_state[2]
    flux[1] = left_state[1]*left_state[2]/left_state[0]+right_state[1]*right_state[2]/right_state[0]
    flux[2] = left_state[2]**2/left_state[0]+PL+right_state[2]**2/right_state[0]+PR
    flux[3] = (left_state[3]+PL)*left_state[2]/left_state[0]+(right_state[3]+PR)*right_state[2]/right_state[0]
    return flux

# def fluxLimiter(state,idx1,idx2,num,den):
#     """This function computers the minmod flux limiter based on pressure ratio"""
#     if (num > 0 and den > 0) or (num < 0 and den < 0):
#         return state[idx1]+min(num/den,1)/2*(state[idx2]-state[idx1])
#     else:
#         return state[idx1]

def fluxLimiter(state,idx1,idx2,num,den):
    """This function computers the minmod flux limiter based on pressure ratio"""
    #Try to form pressure ratio
    Pr = num/den
    Pr = 0 if numpy.isnan(Pr) or numpy.isinf(Pr) or Pr > 1e6 or Pr < 0 else num/den
    tempState = state[idx1]+min(Pr,1)/2*(state[idx2]-state[idx1])
    return tempState

def evaluateSpectral(left_state,right_state,xy):
    """Use this method to compute the Roe Average.
    q(state)
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    spec_state = numpy.zeros(left_state.shape)
    rootrhoL = numpy.sqrt(left_state[0])
    rootrhoR = numpy.sqrt(right_state[0])
    tL = left_state/left_state[0] #Temporary variable to access e, u, v, and w - Left
    tR = right_state/right_state[0] #Temporary variable to access e, u, v, and w -  Right
    #Calculations
    denom = 1/(rootrhoL+rootrhoR)
    spec_state[0] = rootrhoL*rootrhoR
    spec_state[1] = (rootrhoL*tL[1]+rootrhoR*tR[1])*denom
    spec_state[2] = (rootrhoL*tL[2]+rootrhoR*tR[2])*denom
    spec_state[3] = (rootrhoL*tL[3]+rootrhoR*tR[3])*denom
    spvec = (spec_state[0],spec_state[0]*spec_state[1],spec_state[0]*spec_state[2],spec_state[0]*spec_state[3])
    P = getPressure(spvec)
    dim = 1 if xy else 2    #if true provides u dim else provides v dim
    spectralRadius = (numpy.sqrt(gamma*P/spec_state[0])+abs(spec_state[dim]))
    spectralRadius = 0 if numpy.isnan(spectralRadius) else spectralRadius #sets spectral radius to zero if it's nan
    return  spectralRadius*(left_state-right_state)#Returns the spectral radius *(dQ)

def getPressure(q):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-(1/2)*(rho*u^2+rho*v^2))
    """
    return gM1*(q[3]-(q[1]*q[1]+q[2]*q[2])/(2*q[0]))

def getPressureArray(state,iidx):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-(1/2)*(rho*u^2+rho*v^2))
    """
    ops = 2
    pressure = numpy.zeros(state.shape[1:])
    lB = numpy.amin(iidx) #lower bound
    uB = numpy.amax(iidx) #upper bound
    lB = lB-ops if lB>=ops else lB
    uB = uB+ops if uB<(state.shape[-1]-1) else uB #Must adjust this for pseudo 1D debugging
    for i in range(lB,uB+1,1):
        for j in range(lB,uB+1,1): 
            pressure[i,j] = gM1*(state[3,i,j]-(state[1,i,j]*state[1,i,j]+state[2,i,j]*state[2,i,j])/(2*state[0,i,j]))
    return pressure
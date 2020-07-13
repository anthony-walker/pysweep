#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way



import numpy as np
import sympy,sys,itertools
import pycuda.driver as cuda

sfs = "%0.2e"
#Testing
def print_line(line):
    sys.stdout.write('[')
    for item in line:
        sys.stdout.write(sfs%item+", ")
    sys.stdout.write(']\n')
#----------------------------------Globals-------------------------------------#
gamma = 0
dtdx = 0
dtdy = 0
gM1 = 0
#----------------------------------End Globals-------------------------------------#

def step(state,iidx,arrayTimeIndex,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIndex - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    half = 0.5
    ops=2
    vs = slice(0,state.shape[1],1)
    coeff,timechange =  (1,1) if globalTimeStep%2==0 else (0.5,0)  #True - Final Step, False- Intermediate Step
    #Making pressure vector
    l1,l2 = tuple(zip(*iidx))
    l1 = range(min(l1)-ops,max(l1)+ops+1,1)
    l2 = range(min(l2)-ops,max(l2)+ops+1,1)
    P = np.zeros(state[0,0,:,:].shape)

    for idx,idy in itertools.product(l1,l2):
        P[idx,idy] = pressure(state[arrayTimeIndex,vs,idx,idy])

    for idx,idy in iidx:
        dfdx,dfdy = dfdxy(state,(arrayTimeIndex,vs,idx,idy),P)
        state[arrayTimeIndex+1,vs,idx,idy] = state[arrayTimeIndex-timechange,vs,idx,idy]+coeff*(dtdx*dfdx+dtdy*dfdy)
    return state

def set_globals(gpu,*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dtdx,dtdy,gamma,gM1
    t0,tf,dt,dx,dy,gamma = args
    dtdx = dtdy = dt/dx
    if gpu:
        keys = "DT","DX","DY","GAMMA","GAM_M1","DTDX","DTDY"
        nargs = args[2:]+(gamma-1,dt/dx,dt/dy)
        fc = lambda x:np.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))

def createInitialConditions(npx,npy,gamma=1.4,t=0,filename="eulerConditions.hdf5"):
    """Use this function to create a set of initial conditions in an hdf5 file.
    args:
    npx: number of points in x
    npy: number of points in y
    t: time of initial conditions
    """
    # comm = MPI.COMM_WORLD
    # u,X,Y = analytical(npx,npy,t,alpha=alpha)
    # with h5py.File(filename,"w",driver="mpio",comm=comm) as hf:
    #     hf.create_dataset("data",u.shape,data=u)
    # return filename

#---------------------------------------------Solving functions
def dfdxy(state,idx,P):
    """This method is a five point finite volume method in 2D."""
    #Five point finite volume method
    #Creating indices from given point (idx)
    ops = 2 #number of atomic operations
    i1,i2,i3,i4 = idx
    idxx=(i1,i2,slice(i3-ops,i3+ops+1,1),i4)
    idxy=(i1,i2,i3,slice(i4-ops,i4+ops+1,1))
    #Finding spatial derivatives
    dfdx = direction_flux(state[idxx],True,P[idxx[2:]])
    dfdy = direction_flux(state[idxy],False,P[idxy[2:]])
    return dfdx, dfdy

def direction_flux(state,xy,P):
    """Use this method to determine the flux in a particular direction."""
    ONE = 1    #Constant value of 1
    idx = 2     #This is the index of the point in state (stencil data)
    #Initializing Flux
    flux = np.zeros(len(state[:,idx]))

    #Atomic Operation 1
    tsl = flux_limiter(state,idx-1,idx,P[idx]-P[idx-1],P[idx-1]-P[idx-2])
    tsr = flux_limiter(state,idx,idx-1,P[idx]-P[idx-1],P[idx+1]-P[idx])
    flux += eflux(tsl,tsr,xy)
    flux += espectral(tsl,tsr,xy)
    #Atomic Operation 2
    tsl = flux_limiter(state,idx,idx+1,P[idx+1]-P[idx],P[idx]-P[idx-1])
    tsr = flux_limiter(state,idx+1,idx,P[idx+1]-P[idx],P[idx+2]-P[idx+1])
    flux -= eflux(tsl,tsr,xy)
    flux -= espectral(tsl,tsr,xy)
    return flux*0.5

def flux_limiter(state,idx1,idx2,num,den):
    """This function computers the minmod flux limiter based on pressure ratio"""
    dec = 15
    num = round(num,dec)
    den = round(den,dec)
    if (num > 0 and den > 0) or (num < 0 and den < 0):
        return state[:,idx1]+min(num/den,1)/2*(state[:,idx2]-state[:,idx1])
    else:
        return state[:,idx1]

def pressure(q):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-(1/2)*(rho*u^2+rho*v^2))
    """
    return gM1*(q[3]-(q[1]*q[1]+q[2]*q[2])/(2*q[0]))

def eflux(left_state,right_state,xy):
    """Use this method to calculation the flux.
    q (state) is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    #Pressures
    PL = pressure(left_state)
    PR = pressure(right_state)
    #Unpacking
    _,uL,vL,eL = left_state/left_state[0]
    _,uR,vR,eR = right_state/right_state[0]
    #Calculating flux
    #X or Y split
    if xy:  #X flux
        return np.add([left_state[1],left_state[1]*uL+PL,left_state[1]*vL,(left_state[3]+PL)*uL],[right_state[1],right_state[1]*uR+PR,right_state[1]*vR,(right_state[3]+PR)*uR])
    else: #Y flux
        return np.add([left_state[2],left_state[2]*uL,left_state[2]*vL+PL,(left_state[3]+PL)*vL],[right_state[2],right_state[2]*uR,right_state[2]*vR+PR,(right_state[3]+PR)*vR])

def espectral(left_state,right_state,xy):
    """Use this method to compute the Roe Average.
    q(state)
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    # print(left_state,right_state)
    spec_state = np.zeros(len(left_state))
    rootrhoL = np.sqrt(left_state[0])
    rootrhoR = np.sqrt(right_state[0])
    tL = left_state/left_state[0] #Temporary variable to access e, u, v, and w - Left
    tR = right_state/right_state[0] #Temporary variable to access e, u, v, and w -  Right
    #Calculations
    denom = 1/(rootrhoL+rootrhoR)
    spec_state[0] = rootrhoL*rootrhoR
    spec_state[1] = (rootrhoL*tL[1]+rootrhoR*tR[1])*denom
    spec_state[2] = (rootrhoL*tL[2]+rootrhoR*tR[2])*denom
    spec_state[3] = (rootrhoL*tL[3]+rootrhoR*tR[3])*denom
    spvec = (spec_state[0],spec_state[0]*spec_state[1],spec_state[0]*spec_state[2],spec_state[0]*spec_state[3])
    P = pressure(spvec)
    dim = 1 if xy else 2    #if true provides u dim else provides v dim
    return (np.sqrt(gamma*P/spec_state[0])+abs(spec_state[dim]))*(left_state-right_state) #Returns the spectral radius *(dQ)


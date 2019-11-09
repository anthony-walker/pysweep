#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way

import numpy as np
import sys
import itertools
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
#----------------------------------Globals-------------------------------------#
gamma = 0
dtdx = 0
dtdy = 0
gM1 = 0
#----------------------------------End Globals-------------------------------------#

def pm(arr,ps="%0.3f"):
    for item in arr:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write(ps%si+", ")
        sys.stdout.write("]\n")

def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    ops=2
    vs = slice(0,state.shape[1],1)
    sidx = ts-1 if (gts+1)%2==0 else ts #scheme index
    coef = 1 if (gts+1)%2==0 else 0.5 #scheme index
    #Making pressure vector
    l1,l2 = tuple(zip(*iidx))
    l1 = range(min(l1)-ops,max(l1)+ops,1)
    l2 = range(min(l2)-ops,max(l2)+ops,1)
    P = np.zeros(state[0,0,:,:].shape)
    for idx,idy in itertools.product(l1,l2):
        P[idx,idy] = pressure(state[ts,vs,idx,idy])
    for idx,idy in iidx:
        dfdx,dfdy = dfdxy(state,(ts,vs,idx,idy),P)
        state[ts+1,vs,idx,idy] = state[sidx,vs,idx,idy]+coef*(dtdx*dfdx+dtdy*dfdy)
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy,gam = args
    if gpu:
        keys = "DT","DX","DY","GAMMA","GAM_M1","DTDX","DTDY"
        nargs = args[2:]+(gam-1,dt/dx,dt/dy)
        fc = lambda x:np.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        global dtdx
        dtdx = dt/dx
        global dtdy
        dtdy = dt/dy
        global gamma
        gamma = gam
        global gM1
        gM1 = gam-1.0

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

def pressure(q):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-(1/2)*(rho*u^2+rho*v^2))
    """
    riv = 1/(2*q[0])
    return gM1*(q[3]-riv*(q[1]*q[1]+q[2]*q[2]))

def flux_limiter(state,idx1,idx2,ppow,P):
    num = P[idx1+1]-P[idx1]
    den = P[idx1]-P[idx1-1]
    if (num > 0 and den > 0) or (num < 0 and den < 0):
        frac = num/den if ppow else den/num
        return state[:,idx1]+0.5*min(frac,1)*(state[:,idx2]-state[:,idx1])
    else:
        return state[:,idx1]

def direction_flux(state,xy,P):
    """Use this method to determine the flux in a particular direction."""
    ONE = 1    #Constant value of 1
    idx = 2     #This is the index of the point in state (stencil data)
    #Initializing Flux
    flux = np.zeros(len(state[:,idx]))
    #Atomic Operation 1
    tsl = flux_limiter(state,idx-1,idx,True,P)
    tsr = flux_limiter(state,idx,idx-1,False,P)
    flux += eflux(tsl,tsr,xy)
    flux += espectral(tsl,tsr,xy)
    #Atomic Operation 2
    tsl = flux_limiter(state,idx,idx+1,True,P)
    tsr = flux_limiter(state,idx+1,idx,False,P)
    flux -= eflux(tsl,tsr,xy)
    flux -= espectral(tsl,tsr,xy)
    return flux*0.5

def eflux(left_state,right_state,xy):
    """Use this method to calculation the flux.
    q (state) is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    #Initializing Flux
    flux = np.zeros(len(left_state),dtype=np.float32)
    #Pressures
    PL = pressure(left_state)
    PR = pressure(right_state)
    #Unpacking
    _,uL,vL,eL = left_state/left_state[0]
    _,uR,vR,eR = right_state/right_state[0]
    #Calculating flux
    #X or Y split
    if xy:  #X flux
        flux+=[left_state[1],left_state[1]*uL+PL,left_state[1]*vL,(left_state[3]+PL)*uL]
        flux+=[right_state[1],right_state[1]*uR+PR,right_state[1]*vR,(right_state[3]+PR)*uR]
    else: #Y flux
        flux+=[left_state[2],left_state[2]*uL,left_state[2]*vL+PL,(left_state[3]+PL)*vL]
        flux+=[right_state[2],right_state[2]*uR,right_state[2]*vR+PR,(right_state[3]+PR)*vR]
    return flux

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

def test_mod_load():
    """Use this function to test created cu source module."""
    print("Module sucessfully loaded.")

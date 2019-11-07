#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way

import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
#----------------------------------Globals-------------------------------------#
gamma = 0
dtdx = 0
dtdy = 0

#----------------------------------End Globals-------------------------------------#


def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    vs = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        if (gts+1)%2==0:   #Corrector step
            state[ts+1,vs,idx,idy] = state[ts-1,vs,idx,idy]+dtdx*dfdx(state,(ts,vs,idx,idy))#+dtdy*dfdy(state,(ts,vs,idx,idy))
        else: #Predictor step
            state[ts+1,vs,idx,idy] = state[ts,vs,idx,idy]+half*(dtdx*dfdx(state,(ts,vs,idx,idy)))#+dtdy*dfdy)
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

def pressure_ratio(state):
    """Use this function to calculate the pressure ratio for fpfv."""
    #idxs should be in ascending order
    sl = state.shape[1]
    Pr = np.zeros(sl-2)
    pct = 0
    for i in range(1,sl-1):
        try:
            Pr[pct] = ((pressure(state[:,i+1])-pressure(state[:,i]))/
                    (pressure(state[:,i])-pressure(state[:,i-1])))
        except:
            Pr[pct] = np.nan
        pct+=1
    return Pr

def pressure(q):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-rho/2*(rho*u^2+rho*v^2))
    """
    HALF = 0.5
    rho_1_inv = 1/q[0]
    vss = (rho_1_inv*q[1]*q[1])+(rho_1_inv*q[2]*q[2])
    return (gamma - 1)*(q[3]-HALF*vss)

def flimiter(qL,qR,Pr):
    """Use this method to apply the flux limiter to get temporary state."""
    return qL+0.5*min(Pr,1)*(qL-qR) if not np.isinf(Pr) and not np.isnan(Pr) and Pr > 0 else qL

def dfdx(state,inidx):
    """Use this function to determine the x derivative"""
    #Constants
    ops = 2
    ONE = 1    #Constant value of 1
    idx = 2     #This is the index of the point in state (stencil data)
    #Indices
    i0,i1,i2,i3 = inidx #Unpack idxx
    idxx = i0,i1,slice(i2-ops,i2+ops+1,1),i3
    Pr = pressure_ratio(state[idxx])
    dfdx = np.zeros(len(state[:,idx]))

    #Minmod limiter
    tsl = flimiter(state[:,idx-1],state[:,idx],Pr[idx-2])
    tsr = flimiter(state[:,idx],state[:,idx-1],ONE/Pr[idx-1])


    return dfdx

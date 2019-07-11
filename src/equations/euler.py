#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way

import numpy as np
#----------------------------------Globals-------------------------------------#
gamma = 1.4
dtdx = 0.001
dtdy = 0.002
#----------------------------------End Globals-------------------------------------#

def step(state,iidx,ts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx an iterable of indexs
    ts - the current time step (for writing purposes)
    """
    vSlice = slice(0,state.shape[1],1)
    for idx in iidx:
        nidx = (ts+1,vSlice)+idx  #next step index
        idx=(ts,vSlice)+idx  #current step index
        dfdx,dfdy = dfdxy(state,idx)
        state[nidx] += state[idx]+dtdx*dfdx+dtdy*dfdy

def dfdxy(state,idx):
    """This method is a five point finite volume method in 2D."""
    #Five point finite volume method
    #Creating indices from given point (idx)
    ops = 2 #number of atomic operations
    idxx=(idx[0],idx[1],slice(idx[2]-ops,idx[2]+ops+1,1),idx[3])
    idxy=(idx[0],idx[1],idx[2],slice(idx[3]-ops,idx[3]+ops+1,1))

    # print(idxx)
    #Finding pressure ratio
    Prx = pressure_ratio(state[idxx])
    Pry = pressure_ratio(state[idxy])

    #Finding spatial derivatives
    dfdx = direction_flux(state[idxx],Prx,True)
    dfdy = direction_flux(state[idxy],Pry,False)
    return dfdx, dfdy

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
    GAMMA = 1.4
    HALF = 0.5
    rho_1_inv = 1/q[0]
    vss = (q[1]*rho_1_inv)*(q[1]*rho_1_inv)+(q[2]*rho_1_inv)*(q[2]*rho_1_inv)
    return (GAMMA - 1)*(q[3]-HALF*q[0]*vss)

def direction_flux(state,Pr,xy):
    """Use this method to determine the flux in a particular direction."""
    ONE = 1    #Constant value of 1
    idx = 2     #This is the index of the point in state (stencil data)
    #Initializing Flux
    flux = np.zeros(len(state[:,idx]))
    #Atomic Operation 1
    tsl = flimiter(state[:,idx-1],state[:,idx],Pr[idx-2])
    tsr = flimiter(state[:,idx],state[:,idx-1],ONE/Pr[idx-1])
    flux += eflux(tsl,tsr,xy)
    flux += espectral(tsl,tsr,xy)
    #Atomic Operation 2
    tsl = flimiter(state[:,idx],state[:,idx+1],Pr[idx-1])
    tsr = flimiter(state[:,idx+1],state[:,idx],ONE/Pr[idx])
    flux -= eflux(tsl,tsr,xy)
    flux -= espectral(tsl,tsr,xy)
    return flux

def flimiter(qL,qR,Pr):
    """Use this method to apply the flux limiter to get temporary state."""
    return qL+0.5*min(Pr,1)*(qL-qR) if not np.isinf(Pr) and not np.isnan(Pr) and Pr > 0 else qL

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
    #Calculating flux
    #Velocities
    uL = left_state[1]/left_state[0]
    uR = right_state[1]/right_state[0]
    vL = left_state[2]/left_state[0]
    vR = right_state[2]/right_state[0]
    #Pressures
    PL = pressure(left_state)
    PR = pressure(right_state)
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
    spec_state = np.zeros(len(left_state))
    rootrhoL = np.sqrt(left_state[0])
    rootrhoR = np.sqrt(right_state[0])
    tL = left_state/left_state[0] #Temporary variable to access e, u, v, and w - Left
    tR = right_state/right_state[0] #Temporary variable to access e, u, v, and w -  Right

    #Calculations
    denom = 1/(rootrhoL+rootrhoR)
    spec_state[0] += rootrhoL*rootrhoR
    spec_state[1] += (rootrhoL*tL[1]+rootrhoR*tR[1])*denom
    spec_state[2] += (rootrhoL*tL[2]+rootrhoR*tR[2])*denom
    spec_state[3] += (rootrhoL*tL[3]+rootrhoR*tR[3])*denom
    P = pressure(spec_state*spec_state[0])
    dim = 1 if xy else 2    #if true provides u dim else provides v dim
    return (np.sqrt(gamma*P/spec_state[0])+abs(spec_state[dim]))*(left_state-right_state) #Returns the spectral radius *(dQ)

if __name__ == "__main__":
    TA = np.ones((5,4,10,10))
    TA[2,:,5,5] = [3.7,1.4,0.5,0]
    TA[2,:,5,5] = [4.5,2.3,3.8,4]
    TA[2,:,5,5] = [0.526,0.01,0.647,0.328]
    TA[2,:,5,5] = [3.2,1.49,0.25,0.37]
    TA[2,:,5,5] = [0.9,0.7,0.5,0.1]
    # for x in TA:
    #     for y in x:
    #         for t in y:
    #             print(t)
    # Prx = pressure_ratio(TA,((2,5),(3,5),(4,5),(5,5),(6,5)),2)
    step(TA,[(2,5)],2)

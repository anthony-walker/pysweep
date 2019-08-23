#Programmer: Anthony Walker
#These are functions from a validated 1D solver to test functions of the 2D solver

#Import Statements
import numpy as np
import math

mG = 0.4
gamma = 1.4
dtdx = 0.01/0.1

def RK2(Q,P):
    """Use this method to solve a function with RK2 in time."""
    QS = Q+dtdx*0.5*fv5p(Q,P)
    P = eqnStateQ(QS[:,0],QS[:,1],QS[:,2])
    Q = Q+dtdx*fv5p(QS,P)
    return Q

#Numerical Solution
def fv5p(Q,P):
    """Use this 5 point finite volume method to solve the euler equations."""
    #Creating flux array
    Flux = np.zeros((len(Q),len(Q[1,:])))
    #Setting up loop to span entire domain except end points
    for x in range(2,len(Q)-2):
        #Q on the left side of the minus 1/2 interface
        QLm = limiter(Q[x-1],Q[x],Q[x-1],(P[x]-P[x-1]),(P[x-1]-P[x-2]))
        #Q on the right side of the minus 1/2 interface
        QRm = limiter(Q[x],Q[x-1],Q[x],(P[x]-P[x-1]),(P[x+1]-P[x]))
        #Q on the left side of the plus 1/2 interface
        QLp = limiter(Q[x],Q[x+1],Q[x],(P[x+1]-P[x]),(P[x]-P[x-1]))
        #Q on the right side of the plus 1/2 interface
        QRp = limiter(Q[x+1],Q[x],Q[x+1],(P[x+1]-P[x]),(P[x+2]-P[x+1]))
        #Getting Flux
        Flux[x,:] += makeFlux(QLm,QRm)
        Flux[x,:] -= makeFlux(QLp,QRp)
        Flux[x,:] += spectral(QLm,QRm)
        Flux[x,:] -= spectral(QLp,QRp)
        Flux[x,:] = Flux[x,:]*0.5
    return Flux

#Step 2, get reconstructed values of Q at the interfaces
def limiter(Q1,Q2,Q3,num,denom):
    """Use this method to apply the minmod limiter."""
    if (num > 0 and denom > 0) or (num < 0 and denom < 0):
        Q_r = Q1+min(num/denom,1)/2*(Q2-Q3)
    else:
        Q_r = Q1
    return Q_r

#Step 3, make the flux
def makeFlux(QL,QR):
    """Use this method to make Q."""
    uL = QL[1]/QL[0]
    uR = QR[1]/QR[0]
    PL = eqnStateQ(QL[0],QL[1],QL[2])
    FL = np.array([QL[1],QL[1]*uL+PL,(QL[2]+PL)*uL])
    PR = eqnStateQ(QR[0],QR[1],QR[2])
    FR = np.array([QR[1],QR[1]*uR+PR,(QR[2]+PR)*uR])
    return FL+FR

#Step 4, spectral method
def spectral(QL,QR):
    """Use this function to apply the Roe average."""
    Qsp = np.zeros((len(QL)))
    rootrhoL = np.sqrt(QL[0])
    rootrhoR = np.sqrt(QR[0])
    uL = QL[1]/QL[0]
    uR = QR[1]/QR[0]
    eL = QL[2]/QL[0]
    eR = QR[2]/QR[0]
    denom = 1/(rootrhoL+rootrhoR)
    Qsp[0] = (rootrhoL*rootrhoR)
    Qsp[1] = (rootrhoL*uL+rootrhoR*uR)*denom
    Qsp[2] = (rootrhoL*eL+rootrhoR*eR)*denom
    pSP = eqnStateQ(Qsp[0],Qsp[0]*Qsp[1],Qsp[0]*Qsp[2])
    rSP = np.sqrt(gamma*pSP/Qsp[0])+abs(Qsp[1])
    Q_rs = rSP*(QL-QR)
    return Q_rs

#The equation of state using Q variables
def eqnStateQ(r,rU,rE):
    """Use this method to solve for pressure."""
    P = mG*(rE-rU*rU/(r*2))
    return P

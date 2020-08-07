import numpy,sys

### Functions Crucial To Calculation ###
def RK2(state,arrayTimeIndex,globalTimeStep,dt):
    """Use this method to solve a function with RK2 in time."""
    #globals
    global gamma, mG
    gamma = 1.4
    mG = 0.4
    #step data
    dx = 1/(state.shape[-1]-4-1)
    dtdx = dt/dx
    #Creating pressure vector
    if globalTimeStep%2==0:
        P = eqnStateQ(state[arrayTimeIndex,0,:],state[arrayTimeIndex,1,:],state[arrayTimeIndex,3,:])
        state[arrayTimeIndex+1] = state[arrayTimeIndex-1]+dtdx*fv5p(state[arrayTimeIndex],P)   
    else: 
        P = eqnStateQ(state[arrayTimeIndex,0,:],state[arrayTimeIndex,1,:],state[arrayTimeIndex,3,:])
        state[arrayTimeIndex+1] = state[arrayTimeIndex]+dtdx*0.5*fv5p(state[arrayTimeIndex],P)
        
#Numerical Solution
def fv5p(Q,P):
    """Use this 5 point finite volume method to solve the euler equations."""
    #Creating flux array
    Flux = numpy.zeros(Q.shape)
    #Setting up loop to span entire domain except end points
    for x in range(2,Q.shape[-1]-2):
        #Q on the left side of the minus 1/2 interface
        QLm = limiter(Q[:,x-1],Q[:,x],Q[:,x-1],(P[x]-P[x-1]),(P[x-1]-P[x-2]))
        #Q on the right side of the minus 1/2 interface
        QRm = limiter(Q[:,x],Q[:,x-1],Q[:,x],(P[x]-P[x-1]),(P[x+1]-P[x]))
        Flux[:,x] += makeFlux(QLm,QRm)
        Flux[:,x] += spectral(QLm,QRm)

    for x in range(2,Q.shape[-1]-2):
        #Q on the left side of the plus 1/2 interface
        QLp = limiter(Q[:,x],Q[:,x+1],Q[:,x],(P[x+1]-P[x]),(P[x]-P[x-1]))
        #Q on the right side of the plus 1/2 interface
        QRp = limiter(Q[:,x+1],Q[:,x],Q[:,x+1],(P[x+1]-P[x]),(P[x+2]-P[x+1]))
        #Getting Flux
        Flux[:,x] -= makeFlux(QLp,QRp)
        Flux[:,x] -= spectral(QLp,QRp)
        Flux[:,x] = Flux[:,x]*0.5
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
    PL = eqnStateQ(QL[0],QL[1],QL[3])
    FL = numpy.array([QL[1],QL[1]*uL+PL,0,(QL[3]+PL)*uL])
    PR = eqnStateQ(QR[0],QR[1],QR[3])
    FR = numpy.array([QR[1],QR[1]*uR+PR,0,(QR[3]+PR)*uR])
    return FL+FR

#Step 4, spectral method
def spectral(QL,QR):
    """Use this function to apply the Roe average."""
    Qsp = numpy.zeros((len(QL)))
    rootrhoL = numpy.sqrt(QL[0])
    rootrhoR = numpy.sqrt(QR[0])
    uL = QL[1]/QL[0]
    uR = QR[1]/QR[0]
    eL = QL[3]/QL[0]
    eR = QR[3]/QR[0]
    denom = 1/(rootrhoL+rootrhoR)
    Qsp[0] = (rootrhoL*rootrhoR)
    Qsp[1] = (rootrhoL*uL+rootrhoR*uR)*denom
    Qsp[3] = (rootrhoL*eL+rootrhoR*eR)*denom
    pSP = eqnStateQ(Qsp[0],Qsp[0]*Qsp[1],Qsp[0]*Qsp[3])
    rSP = numpy.sqrt(gamma*pSP/Qsp[0])+abs(Qsp[1])
    Q_rs = rSP*(QL-QR)
    return Q_rs

#The equation of state using Q variables
def eqnStateQ(r,rU,rE):
    """Use this method to solve for pressure."""
    global mG
    P = mG*(rE-rU*rU/(r*2))
    return P
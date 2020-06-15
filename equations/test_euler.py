import numpy as np
import sys
gamma = 1.4
mG = gamma-1
dt = dx = dtdx =  None

sfs = "%0.2e"
def print_line(line):
    sys.stdout.write('[')
    for item in line:
        sys.stdout.write(sfs%item+", ")
    sys.stdout.write(']\n')


def step(Q,QS,ts):
    """Use this method to solve a function with RK2 in time."""
    #Preliminary Steps
    # print(ts,'1D')
    if (ts+1)%2 == 0: #Corrector
        # print_line(QS[0,:])
        P = eqnStateQ(QS[0,:],QS[1,:],QS[2,:])
        # print_line(P)
        # sys.stdout.write('[')
        Q = Q+dtdx*fv5p(QS,P)
        # sys.stdout.write(']\n')
        return Q
    else: #Predictor
        # print_line(Q[0,:])
        P = eqnStateQ(Q[0,:],Q[1,:],Q[2,:])
        # print_line(P)
        # sys.stdout.write('[')
        QS = Q+dtdx*0.5*fv5p(Q,P)
        # sys.stdout.write(']\n')
        return QS

def set_globals(gam, dtl,dxl):
    """Temp set globals function"""
    global dt,dx,gamma,mG,dtdx
    gamma = gam
    dt = dtl
    dx = dxl
    dtdx = dt/dx
    mG = gam-1

#Numerical Solution
def fv5p(Q,P):
    """Use this 5 point finite volume method to solve the euler equations."""
    #Creating flux array
    Flux = np.zeros(Q.shape)
    #Setting up loop to span entire domain except end points
    for x in range(2,Q.shape[1]-2,1):
        pb = True if x == 11 else False
        #Q on the left side of the minus 1/2 interface
        QLm = limiter(Q[:,x-1],Q[:,x],Q[:,x-1],(P[x]-P[x-1]),(P[x-1]-P[x-2]),True)
        # sys.stdout.write(sfs%QLm[0]+", ")
        #Q on the right side of the minus 1/2 interface
        QRm = limiter(Q[:,x],Q[:,x-1],Q[:,x],(P[x]-P[x-1]),(P[x+1]-P[x]),False)
        # sys.stdout.write(sfs%QRm[0]+", ")
        #Q on the left side of the plus 1/2 interface
        QLp = limiter(Q[:,x],Q[:,x+1],Q[:,x],(P[x+1]-P[x]),(P[x]-P[x-1]),False)
        #Q on the right side of the plus 1/2 interface
        QRp = limiter(Q[:,x+1],Q[:,x],Q[:,x+1],(P[x+1]-P[x]),(P[x+2]-P[x+1]),False)
        #Getting Flux
        Flux[:,x] += makeFlux(QLm,QRm)
        # sys.stdout.write(sfs%Flux[0,x]+", ")
        Flux[:,x] -= makeFlux(QLp,QRp)
        Flux[:,x] += spectral(QLm,QRm)
        Flux[:,x] -= spectral(QLp,QRp)
        Flux[:,x] *= 0.5
    return Flux

#Step 1, make Q from node data
def makeQ(Nargs):
    """Use this method to make Q."""
    dim1 = len(Nargs)
    dim2 = len(Nargs[1,:])
    Q = np.zeros((dim1,dim2))
    k = 0
    for i in Nargs:
        Q[k,:] = np.array([i[0],i[1]*i[0],i[0]*i[2]])
        k +=1
    return Q

#Step 2, get reconstructed values of Q at the interfaces
def limiter(Q1,Q2,Q3,num,den,pb):
    """Use this method to apply the minmod limiter."""
    # if pb:
        # sys.stdout.write( "C0:"+str(den>0)+" C1:"+str(den<0)+", ")
        # sys.stdout.write( str(den>0)+", ")
        # sys.stdout.write("%0.1e"%den+", ")
        # sys.stdout.write(str(type(den))+", ")
    dec = 15
    num = round(num,dec)
    den = round(den,dec)
    if (num > 0 and den > 0) or (num < 0 and den < 0):
        # sys.stdout.write('A, ')
        # if pb:
        #     sys.stdout.write("N:"+sfs%num+", ")
        #     sys.stdout.write("D:"+sfs%den+", ")
        Q_r = Q1+min(num/den,1)/2*(Q2-Q3)
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

if __name__ == "__main__":
    def testOneDimEuler():
        """Use this function as a test for this file"""
        set_globals(1.4,0.1,0.1)
    testOneDimEuler()

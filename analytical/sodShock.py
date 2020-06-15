#Programmer: Anthony Walker
import numpy as np
import datetime as dt
import scipy.optimize
import h5py, os
#Allocation of global variables
gamma = 0
alpha = 0
beta = 0
cL = 0
cR = 0
lbc = 0
rbc = 0
t = 0
numPts = 0
epsilon = 0

def sodShock(time,npts,xy,fileName='sodshock.hdf5',lBC = (1.0,1.0,0,2.5),rBC = (0.1,0.125,0,2),g = 1.4):
    """Use this method to determine the analytic solution of sod shock problem."""
    #Assigning Global variables
    fp = os.path.abspath(__file__)[:-len(os.path.basename(__file__))-len('analytical/')]
    fp = os.path.join(os.path.join(os.path.join(fp,'tests'),'data'),fileName)
    gVI(g,rBC,lBC,time,npts)
    #Determining states
    statesList = states2and3()
    rho = statesList[0]
    u = statesList[1]
    p = statesList[2]
    speed = statesList[3]
    #Location
    x1 = 0
    xshock = x1+speed[3]*t
    xcontact = x1 +speed[2]*t
    xfR = x1 + speed[1]*t
    xfL = x1 + speed[0]*t
    x = [xfL,xfR,x1,xcontact,xshock]

    #Getting Data
    Xloc = 0.5
    xI = np.linspace(-Xloc,Xloc,numPts)
    pD = np.zeros(numPts)
    rhoD = np.zeros(numPts)
    uD = np.zeros(numPts)
    arr = np.zeros((4,npts))
    combList = list()
    for i in range(numPts): #State L
        if(xI[i]<xfL):
            pD[i] = lbc[0]
            rhoD[i] = lbc[1]
            uD[i] = lbc[2]
        elif(xI[i] < xfR and xI[i] > xfL): #State 4
            pD[i] = lbc[0]*(1+(xfL-xI[i])/(cL*alpha*t))**(beta)
            uD[i] = lbc[2]-2/(gamma+1)*(xfL-xI[i])/t
            rhoD[i] = lbc[1]*(1-(gamma-1)*uD[i]/(2*cL))**(2/(gamma-1))
        elif(xI[i] < xcontact and xI[i] > xfR): #State 3
            rhoD[i] = rho[1]
            pD[i] = p[1]
            uD[i] = u[1]
        elif(xI[i] < xshock and xI[i] > xcontact): #State 2
            rhoD[i] = rho[0]
            pD[i] = p[0]
            uD[i] = u[0]
        elif(xI[i] > xshock):
            pD[i] = rbc[0]
            rhoD[i] = rbc[1]
            uD[i] = rbc[2]
        eTemp = eqnState(pD[i],rhoD[i],uD[i],gamma)
        if xy:
            arr[:,i] = [rhoD[i],rhoD[i]*uD[i],0,rhoD[i]*eTemp]
        else:
            arr[:,i] = [rhoD[i],0,rhoD[i]*uD[i],rhoD[i]*eTemp]
    # arr = arr if xy else np.transpose(arr)
    return arr

def eqnState(p,rho,u,gamma):
    """Use this method to solve for pressure."""
    e = p/(gamma-1)+rho*u**2/2
    return e

def states2and3(tol = (10**-10)):
    "Use this method to solve for the states"
    # Left side conditions
    pL = lbc[0]
    rhoL = lbc[1]
    uL = lbc[2]
    # Right side conditions
    pR = rbc[0]
    rhoR = rbc[1]
    uR = rbc[2]
    #iteratively solving Rankine-Hugoniot relation for P2
    solBool = False
    for pG in np.linspace(pR, pL, numPts):
        if(not solBool):
            rhSol = scipy.optimize.fsolve(RankineHugoniot, pG, full_output=True)
            p2, infodict, solBool, mesg = rhSol
            if type(p2) is np.ndarray:
                p2 = p2[0]
        else:
            break
    #State 2
    P = (p2/pR-1)
    temp2 = np.sqrt(1 + P/epsilon)
    rho2 = rhoR*(1+P/epsilon)/(1+P/beta)
    u2 = cR*P/(gamma*temp2)
    ss =  cR*temp2 #Shock speed

    #State 3
    p3 = p2
    u3 = u2
    rho3 = rhoL*(p3/pL)**(1/gamma)
    sc = u3 #Contact speed

    #State 4
    sfR = u3-np.sqrt(gamma*p3/rho3) #Expansion fan right speed
    sfL = -cL #Expansion fan left speed


    #Return variables
    speed =[sfL,sfR,sc,ss]
    rho = [rho2,rho3]
    u = [u2,u3]
    p = [p2,p3]
    return [rho, u, p,speed]

def RankineHugoniot(P):
    """Use this method to solve the Rankine-Hugoniot relation."""
    # prevP = P+1
    # while(abs(P-prevP)>tol):
    # prevP = P
    Pr = P/rbc[0]
    temp = 1/beta*(cR/cL)*(Pr-1)
    temp = temp/np.sqrt(1+(gamma+1)/(2*gamma)*(Pr-1))
    temp = (1-temp)**beta
    # return P
    return temp*lbc[0]-P

def gVI(g,rBC,lBC,time,npts):
    """Use this method to initialize global variables."""
    #Important coeffcients
    global gamma
    gamma = g
    global alpha
    alpha = (gamma+1)/(gamma-1)
    global beta
    beta = (2*gamma)/(gamma-1)
    global epsilon
    epsilon = (2*gamma)/(gamma+1)
    #Boundary conditions
    global lbc
    lbc = lBC
    global rbc
    rbc = rBC
    #Time
    global t
    t = time
    #points
    global numPts
    numPts = npts
    #Speed of sound for states 1 and 5
    global cL
    cL = np.sqrt(gamma*lbc[0]/lbc[1])
    global cR
    cR = np.sqrt(gamma*rbc[0]/rbc[1])

if __name__ == "__main__":
    pass

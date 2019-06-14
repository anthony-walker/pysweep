#Programmer: Anthony Walker
#README:
#This is a function specifically built to solve two dimensional euler equations

def fpfv(D):
    #Five point finite volume method
    Flux = np.zeros(D.shape)
    #Interior points
    for i in D.pts:
        #Getting all points for stencil
        cDim = 0 #Current Dimension - starts at 2 for make flux
        pts = D.stencils[i]
        for p in pts:
            F = np.zeros(D.shape[-1])
            WW = D.primaryDomain[p[0]]
            W = D.primaryDomain[p[1]]
            P = D.primaryDomain[i]
            E = D.primaryDomain[p[2]]
            EE = D.primaryDomain[p[3]]
            #Step 1, Applying minmod limiter functiongamma
            #Q on the left side of the minus 1/2 interface
            QLm = limiter(W,P,W,(P[0]-W[0]),(W[0]-WW[0]))
            #Q on the right side of the minus 1/2 interface
            QRm = limiter(P,W,P,(P[0]-W[0]),(E[0]-P[0]))
            #Q on the left side of the plus 1/2 interface
            QLp = limiter(P,E,P,(E[0]-P[0]),(P[0]-W[0]))
            #Q on the right side of the plus 1/2 interface
            QRp = limiter(E,P,E,(E[0]-P[0]),(EE[0]-E[0]))
            #Step 2, Make Flux with reconstructed values
            F += makeFlux(QLm,QRm,cDim,D.gamma)
            F -= makeFlux(QLp,QRp,cDim,D.gamma)
            #Roe Average Here
            F += roeAvg(QLm,QRm,cDim,D.gamma)*(QLm-QRm)
            F -= roeAvg(QLp,QRp,cDim,D.gamma)*(QLp-QRp)
            Flux[i] += F/D.stepsizes[cDim]
            cDim +=1
    return Flux*(0.5)

#Step 1 - Minmod limiter function
def limiter(Q1,Q2,Q3,num,denom):
    """Use this method to apply the minmod limiter."""
    if (num > 0 and denom > 0) or (num < 0 and denom < 0):
        Q_r = Q1+min(num/denom,1)/2*(Q2-Q3)
    else:
        Q_r = Q1
    return Q_r

#Step 2 - Making Flux
def makeFlux(QL,QR,dim,gamma):
    """Use this method to make Q."""
    #Q = (P,rho, rhoe, rhou, rhov, rhow)
    #P is not calculated here. It is just a place holder to maintain shape for addition
    # dim = 3, 4, 5 for x, y, and z
    dim = dim+3 #rho*u is has index 3. So, to get appropriate dimension 3 is added
    #Preallocation
    FL = np.zeros(len(QL))
    FR = np.zeros(len(QR))
    #Find pressure - equation of state
    #(gamma-1)*(rho*e-rho/2*(u^2+v^2+w^2))
    PL = (gamma - 1)*(QL[2]-QL[1]/2*((QL[3]/QL[1])**2+(QL[4]/QL[1])**2
                    +(QL[5]/QL[1])**2))
    PR = (gamma - 1)*(QR[2]-QR[1]/2*((QR[3]/QR[1])**2+(QR[4]/QR[1])**2
                    +(QR[5]/QR[1])**2))
    #Part 1 - rho*(u || v || w)
    FL[1] = QL[dim]
    FR[1] = QR[dim]

    #Part 2 - (rho*e+P)*rho*(u || v || w)/rho
    FL[2] = (QL[2]+PL)*QL[dim]/QL[1]
    FR[2] = (QR[2]+PR)*QR[dim]/QR[1]

    #Part 3-5 - Left rho*(u || v || w)*rho(x && y && z)
    FL[3] = QL[dim]*QL[3]/QL[1]
    FL[4] = QL[dim]*QL[4]/QL[1]
    FL[5] = QL[dim]*QL[5]/QL[1]

    #Part 3-5 - Right
    FR[3] = QR[dim]*QR[3]/QR[1]
    FR[4] = QR[dim]*QR[4]/QR[1]
    FR[5] = QR[dim]*QR[5]/QR[1]

    #Variable part - this will overwrite the incorrect value in parts 3-5
    FL[dim] = QL[dim]*QL[dim]/QL[1]+PL
    FR[dim] = QR[dim]*QL[dim]/QL[1]+PR

    #Return sum of two sides
    return FL+FR

#Step 3 - Roe Average - ***CHECK ME***
def roeAvg(QL,QR,dim,gamma):
    """Use this method to compute the Roe Average."""
    Qsp = np.zeros(len(QL))
    rootrhoL = np.sqrt(QL[1])
    rootrhoR = np.sqrt(QR[1])
    tL = QL/QL[1] #Temporary variable to access e, u, v, and w - Left
    tR = QR/QR[1] #Temporary variable to access e, u, v, and w -  Right
    #Calculations
    denom = 1/(rootrhoL+rootrhoR)
    Qsp[1] += rootrhoL*rootrhoR
    Qsp[2] += (rootrhoL*tL[2]+rootrhoR*tR[2])*denom
    Qsp[3] += (rootrhoL*tL[3]+rootrhoR*tR[3])*denom
    Qsp[4] += (rootrhoL*tL[4]+rootrhoR*tR[4])*denom
    Qsp[5] += (rootrhoL*tL[5]+rootrhoR*tR[5])*denom
    pSP = (gamma-1)*(Qsp[1]*Qsp[2]-Qsp[1]/2*(Qsp[3]**2+Qsp[4]**2+Qsp[5]**2))
    rSP = np.sqrt(gamma*pSP/Qsp[1])+abs(Qsp[dim])
    return rSP #Returns the spectral radius

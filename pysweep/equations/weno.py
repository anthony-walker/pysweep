import numpy

def testConditions(state):
    """Use this function to generate the Weno Scheme 2D Test Conditions."""

    #Closure to wrap numpy fill for array like value
    def nfill(values,arrayToFill):
        """Use this function to fill arrayToFill with values array."""
        for i,val in enumerate(values):
            arrayToFill[i].fill(val)

    nv,nx,ny = state.shape
    yhalf = int(ny//2)
    xhalf = int(nx//2)
    #Quadrant 1
    nfill([1.5,0,0,1.206],state[:,:xhalf,:yhalf])
    #Quadrant 2
    nfill([0.5323,1.206,0,0.3],state[:,xhalf:,:yhalf])
    #Quadrant 3
    nfill([0.138,1.206,1.206,0.029],state[:,:xhalf,yhalf:])
    #Quadrant 4
    nfill([0.5323,0,1.206,0.3],state[:,xhalf:,yhalf:])  

 def wenoFlux(spectralRadius,flux,spatialStep,direction):
     """Use this to get the reconstructed flux in a particular direction."""
     

if __name__ == "__main__":
    gamma = 1.4
    gammaMinusOne = gamma-1
    stateVector = numpy.zeros((4,6,6)) #Creating initial flux vector
    testConditions(stateVector)  #Filling vector with test conditions
    P = stateVector[-1,:,:]#Get pressure
    stateVector[-1] = P/(gammaMinusOne*stateVector[0])+0.5*(stateVector[1]**2+stateVector[2]**2) #Replace pressure with total energy density
    speedOfSound = numpy.sqrt(gamma*P/stateVector[0])
    #making flux vector
    flux = numpy.zeros(stateVector.shape)
    flux[0,:,:] = stateVector[0,:,:]
    flux[1,:,:] = stateVector[0,:,:]*stateVector[1,:,:]
    flux[2,:,:] = stateVector[0,:,:]*stateVector[2,:,:]
    flux[3,:,:] = stateVector[0,:,:]*stateVector[3,:,:]
    
    #Set initial conditions
    ctr=0 #iteration counter
    currentTime=0 #currentTime


    #Using systems largest eigen value
    
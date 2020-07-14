
import numpy, h5py, os, sys
import mpi4py.MPI as MPI
try:
    import pycuda.driver as cuda
except Exception as e:
    print(e)
#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib import animation, rc

#globals
scheme = True
pi = numpy.pi
piSq = pi*pi

def step1D(state):
    """This is the method that will be called by the swept solver.
    state - 3D numpy array(t,v,x (v is variables length))
    arrayTimeIndex - array time index for state
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    nx = state.shape[-1]
    stateStar = numpy.zeros(state.shape[-1])
    c = alpha*dt/dx/dx
    for idx in range(nx):
        #Predictor
        stateStar[idx] = state[0,0,idx]+c*centralDifference(state[0,0],idx)
        
    for idx in range(nx):
        #Corrector
        state[1,0,idx] = state[0,0,idx]+0.5*c*(centralDifference(stateStar,idx)+centralDifference(state[0,0],idx))

def centralDifference(state,idx):
    nx = state.shape[-1]
    # print((idx+1)%nx,idx,idx-1)
    return state[(idx+1)%nx]-2*state[idx]+state[idx-1]


def getAnalytical1D(npx,t,alpha):
    T = numpy.zeros((1,npx))
    X = numpy.linspace(0,1,npx,endpoint=False)
    for i,x in enumerate(X):
        T[0,i] = numpy.sin(2*pi*x)*numpy.exp(-4*pi*pi*alpha*t)
    return T

def verifyOrder1D(fname=None):
    """Use this function to verify the order of Runge-Kutta Second order."""
    global dt,dx,dy,alpha,courant
    courant = 0.1
    sizes = [10,100]#[int(i*12) for i in range(1,3)]
    error = []
    dtCases = []
    tf = 0.1
    for arraysize in sizes:
        alpha = 1
        dx = 1/arraysize
        dt = float(courant*dx*dx/alpha)
        timesteps = int(tf//dt)
        dtCases.append(dt)
        T = getAnalytical1D(arraysize,0,alpha) #get ICs
        numerical = numpy.zeros((2,)+T.shape)
        numerical[0,0,:] = T[0,:]
        for i in range(timesteps): #current time step
            step1D(numerical)
            numerical[0,:,:] = numerical[1,:,:]
        T = getAnalytical1D(arraysize,tf,alpha)
        error.append(numpy.amax(numpy.absolute(numerical[-1,0,:]-T[0,:])))   
    print((numpy.log(error[1])-numpy.log(error[0]))/(numpy.log(dtCases[1])-numpy.log(dtCases[0])))
    
    plt.loglog(dtCases,error,marker='o')
    # plt.ylim(10e-6,10e-4)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

if __name__ == "__main__":
    verifyOrder1D()
#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
from src.sweep import *
from src.analytical import *
from src.equations import *

#Properties
gamma = 1.4
times = np.linspace(1,5,10)

#Analytical properties
avics = vics()  #Creating vortex ics object
avics.Shu(gamma) #Initializing with Shu parameters

#Calling analytical solution
def analytical():
    """Use this funciton to solve the analytical euler vortex."""
    create_vortex_data(cvics,X,Y,npx,npy,times=(0,0.1))

def swept():
    """Use this function to execute the swept rule."""
    # print("Starting execution.")
    dims = (4,int(256),int(256))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 0.1
    arr0[1,:,:] = 0.5
    arr0[2,:,:] = 0.2
    arr0[3,:,:] = 0.125

    #GPU Arguments
    block_size = (32,32,1)
    kernel = "/home/walkanth/pysweep/src/equations/euler.h"
    affinity = 0.8
    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.01
    targs = (t0,t_b,dt)

    #Space testing arguments
    dx = 0.1
    dy = 0.1
    ops = 2

    sweep(arr0,targs,dx,dy,ops,block_size,kernel,affinity)


if __name__ == "__main__":
    swept()

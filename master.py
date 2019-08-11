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
    dims = (4,int(16),int(16))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 2
    arr0[1,:,:] = 1
    arr0[2,:,:] = 1
    arr0[3,:,:] = 9

    #GPU Arguments
    block_size = (8,8,1)
    kernel = "/home/walkanth/pysweep/src/equations/euler.h"
    cpu_source = "/home/walkanth/pysweep/src/equations/euler.py"
    affinity = 0.5    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.1
    targs = (t0,t_b,dt)

    #Space testing arguments
    dx = 0.1
    dy = 0.1
    ops = 2

    sweep(arr0,targs,dx,dy,ops,block_size,kernel,cpu_source,affinity,filename="./results/solution")


if __name__ == "__main__":
    swept()

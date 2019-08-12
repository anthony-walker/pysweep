#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
from src.sweep import *
from src.analytical import *
from src.equations import *
from src.decomp import *

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

def create_test_args():
    """Use this function to create arguements for the two codes."""
    for i in range(6,34,2):
        # print(i)
        pass

def create_sets(block_size,ops):
    """Use this function to create index sets."""
    bsx = block_size[0]+2*ops
    bsy = block_size[1]+2*ops
    ly = ops
    uy = bsy-ops
    min_bs = int(min(bsx,bsy)/(2*ops))
    iidx = tuple(np.ndindex((bsx,bsy)))
    idx_sets = tuple()
    for i in range(min_bs):
        iidx = iidx[ops*(bsy-i*2*ops):-ops*(bsy-i*2*ops)]
        iidx = [(x,y) for x,y in iidx if y >= ly and y < uy]
        if len(iidx)>0:
            idx_sets+=(iidx,)
        ly+=ops
        uy-=ops
    return idx_sets


if __name__ == "__main__":
    dims = (4,int(16),int(16))
    arr0 = np.zeros(dims)
    arr0[0,:,:] = 1
    arr0[1,:,:] = 2
    arr0[2,:,:] = 3
    arr0[3,:,:] = 4

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
    # idx = create_sets(block_size,ops)
    # print(idx[1])
    # print(len(idx[1]))
    # print(len(idx))
    sweep(arr0,targs,dx,dy,ops,block_size,kernel,cpu_source,affinity,filename="./results/swept")
    # decomp(arr0,targs,dx,dy,ops,block_size,kernel,cpu_source,affinity,filename="./results/decomp")

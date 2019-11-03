from pysweep.sweep.distsweep import dsweep
import numpy as np
from pysweep.analytical.ahde import TIC


if __name__ == "__main__":
    # nx = ny = 120
    ny = 120
    nx = 120
    bs = 12
    t0 = 0
    gamma = 1.4
    # arr = np.ones((4,nx,ny))

    # ct = 0
    # for i in range(nx):
    #     for j in range(ny):
    #         arr[0,i,j] = ct
    #         ct+=1
    # arr[:,256:,:] += 1
    X = 10
    Y = 10
    tso = 2
    ops = 1
    aff = 0 #Dimensions and steps
    dx = X/nx
    dy = Y/ny
    alpha = 5
    Fo = 0.24
    dt = Fo*(X/nx)**2/alpha
    tf = dt*100
    arr = TIC(nx,ny,X,Y,1,373,298)[0,:,:,:]
    #Changing arguments
    gargs = (t0,tf,dt,dx,dy,alpha)
    swargs = (tso,ops,bs,aff,"./pysweep/equations/hde.h","./pysweep/equations/hde.py")
    dsweep(arr,gargs,swargs,filename="test",exid=[])

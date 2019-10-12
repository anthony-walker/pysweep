from pysweep.sweep.dist_sweep import dsweep
import numpy as np

if __name__ == "__main__":
    nx = ny = 32
    bs = 8
    t0 = 0
    tf = 0.1
    dt = 0.01
    dx = dy = 0.1
    gamma = 1.4
    arr = np.zeros((4,nx,ny))
    # ct = 0
    # for i in range(nx):
    #     for j in range(ny):
    #         arr[0,i,j] = ct
    #         ct+=1
    # arr[:,256:,:] += 1
    X = 1
    Y = 1
    tso = 2
    ops = 2
    aff = 0.5 #Dimensions and steps
    dx = X/nx
    dy = Y/ny
    #Changing arguments
    gargs = (t0,tf,dt,dx,dy,gamma)
    swargs = (tso,ops,bs,aff,"./src/equations/eqt.h","./src/equations/eqt.py")
    dsweep(arr,gargs,swargs,filename="test",exid=[])

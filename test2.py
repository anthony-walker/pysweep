from src.sweep import dsweep
if __name__ == "__main__":
    nx = ny = 512
    bs = 8
    t0 = 0
    tf = 0.1
    dt = 0.01
    dx = dy = 0.1
    gamma = 1.4
    arr = np.ones((4,nx,ny))
    X = 1
    Y = 1
    tso = 2
    ops = 2
    aff = 0.5    #Dimensions and steps
    dx = X/nx
    dy = Y/ny
    #Changing arguments
    gargs = (t0,tf,dt,dx,dy,gamma)
    swargs = (tso,ops,bs,aff,"./src/equations/euler.h","./src/equations/euler.py")
    dsweep(arr,gargs,swargs,filename="test")

#Programmer: Anthony Walker
#Analytical solution to the unsteady 2D heat equation with periodic boundary conditions

import numpy as np
import h5py

def AHDE(L,W,nx,ny,R=1,times=(0,),Th = 373,Tl = 298,alpha=25e-6,dType=np.dtype('float32'),filepath="./tests/data/temp.hdf5"):
    """Analytical Heat Diffusion Equation"""

    #Points
    xpts = np.linspace(0,L,nx,dtype=dType)
    ypts = np.linspace(0,W,ny,dtype=dType)
    T = np.zeros((len(times),1,nx,ny))
    #Fourier Coefficients
    mm = nm = 10
    Amn_c = lambda x,y: 4*Th/(L*W) if x <= L/2+R and x >= L/2-R and y <= W/2+R and y >= W/2-R else 4*Tl/(L*W)
    bounds = (0,L/2-R,0,W/2-R),(0,L/2-R,W/2-R,W/2+R),(0,L/2-R,W/2+R,W)
    bounds += (L/2-R,L/2+R,0,W/2-R),(L/2-R,L/2+R,W/2-R,W/2+R),(L/2-R,L/2+R,W/2+R,W)
    bounds += (L/2+R,L,0,W/2-R),(L/2+R,L,W/2-R,W/2+R),(L/2+R,L,W/2+R,W)

    int_sum=0
    fcns = tuple()
    for bset in bounds:
        x1,y1,x2,y2 = bset
        fcns += (lambda m,n,x,y: Amn_c(x,y)*L/(n*np.pi)*(np.cos(n*np.pi*x1/L)-np.cos(n*np.pi*x2/L))*W/(m*np.pi)*(np.cos(m*np.pi*y1/W)-np.cos(m*np.pi*y2/W))),
    m = 1
    n = 1
    x = 1
    y = 1

    transient = lambda m,n,t: np.exp(-alpha*((m*np.pi/W)**2+(n*np.pi/L)**2)*t)
    fourier = lambda x,y,m,n,Amn: Amn*np.sin(n*np.pi*x/L)*np.sin(m*np.pi*y/W)


    for idt,t in enumerate(times):
        for idx,x in enumerate(xpts):
            for idy,y in enumerate(ypts):
                fr = 0
                for m in range(1,mm+1):
                    for n in range(1,nm+1):
                        Amn = 0
                        for f in fcns:
                            Amn+=f(m,n,x,y)
                        fr += fourier(x,y,m,n,Amn)+transient(m,n,t)
                T[idt,0,idx,idy] = fr

    file = h5py.File(filepath,"w")
    T_data = file.create_dataset("data",T.shape)
    dim_data = file.create_dataset("dimensions",(4,))
    a_data = file.create_dataset("alpha",(1,))
    t_data = file.create_dataset("time",(len(times),))
    T_data[:,:,:,:] = T[:,:,:,:]
    dim_data[:] = (nx,ny,L,W)
    a_data[:] = alpha
    t_data[:] = times[:]
    file.close()


def TIC(nx,ny,X,Y,R,Th,Tl,lt = 1):
    """Use this function to generate the initical conditions array"""
    T = Tl*np.ones((lt,1,nx,ny))
    for idx,x in enumerate(np.linspace(0,X,nx)):
        for idy,y in enumerate(np.linspace(0,Y,ny)):
            if x <= X/2+R and x >= X/2-R and y <= Y/2+R and y >= Y/2-R:
                T[0,0,idx,idy] = Th
    return T

if __name__ == "__main__":
    AHDE(10,10,100,100)

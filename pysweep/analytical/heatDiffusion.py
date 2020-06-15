#Programmer: Anthony Walker
#This file contains an implementation of a 2D heat diffusion equation analytical solution

#imports
import numpy as np

class HeatDiffusion(object):
    """This class contains an implementation of a specific analytical solution for the
       two dimensional heat diffusion equation."""

    def __init__(self, size,c):
        """This is the constructor
        Parameters:
        size - number of points (n) in an n x n matrix.
        """
        super(HeatDiffusion, self).__init__()
        self.size = size
        self.c = c
        self.L = np.linspace(-2,2,size)
        self.mesh = np.zeros((size,size,2))
        for i in range(size):
            for j in range(size):
                self.mesh[i,j,:] = self.L[i],self.L[j]
        self.solution()

    def solution(self,t=0):
        self.sol = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(self.size):
                x,y = self.mesh[i,j]
                self.sol[i,j] = np.sin(np.pi*x)*np.sin(np.pi*y)+np.sin(np.pi*x)*np.cos(np.pi*y)
                self.sol[i,j] += np.cos(np.pi*x)*np.cos(np.pi*y)+np.cos(np.pi*x)*np.sin(np.pi*y)
                self.sol[i,j] *= np.exp(-2*np.pi*self.c*t)
        return self.sol

def test_heatDiffusion():
    c = 1
    s = 100
    hObject = HeatDiffusion(s,c)
    dx = abs(hObject.L[1]-hObject.L[0])
    dt = 0.0001
    nSol = np.copy(hObject.sol)
    idx = np.ndindex(nSol.shape)
    #Numerical implementation
    t = 0
    for k in range(2):
        bool = np.allclose(nSol,hObject.solution(t))
        print(bool)
        #Interior
        for i in range(s):
            for j in range(s):
                b1 = (i%99)
                b2 = (j%99)
                if b1 and b2:
                    temp = c*dt/(dx*dx)*(nSol[(i-1),j]-2*nSol[i,j]+nSol[(i+1),j]) #uxx
                    temp += c*dt/(dx*dx)*(nSol[i,(j-1)]-2*nSol[i,j]+nSol[i,(j+1)]) #uyy
                elif not b1 and b2:
                    temp = c*dt/(dx*dx)*(nSol[-2,j]-2*nSol[i,j]+nSol[1,j]) #uxx
                    temp += c*dt/(dx*dx)*(nSol[i,(j-1)]-2*nSol[i,j]+nSol[i,(j+1)]) #uyy
                elif b1 and not b2 :
                    temp = c*dt/(dx*dx)*(nSol[(i-1),j]-2*nSol[i,j]+nSol[(i+1),j]) #uxx
                    temp += c*dt/(dx*dx)*(nSol[i,-2]-2*nSol[i,j]+nSol[i,1]) #uyy
                else:
                    temp = c*dt/(dx*dx)*(nSol[-2,j]-2*nSol[i,j]+nSol[1,j]) #uxx
                    temp += c*dt/(dx*dx)*(nSol[i,-2]-2*nSol[i,j]+nSol[i,1]) #uyy
                nSol[i,j] += temp
        t+=dt

if __name__ == "__main__":
    test_heatDiffusion()

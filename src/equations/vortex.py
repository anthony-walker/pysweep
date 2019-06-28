#Programmer: Anthony Walker
#This file contains the analytical solution to the euler vortex
#problem in two dimensions.
import numpy as np
from vortex_conditions import *

def vortex(args):
    """This is the primary method to solve the euler vortex."""
    alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, sigma, beta, R, L = args #unpacking args
    npts = 2*L/R
    x = np.linspace(-L,L,npts,dtype=np.float64)
    y = np.linspace(-L,L,npts,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(x,y,sparse=False,indexing='ij')
    f = np.zeros(xgrid.shape,dtype=np.float64)
    f = -1/(2*sigma**2)*((xgrid/R)**2+(ygrid/R)**2)
    #Gaussian function
    Omega = beta*np.exp(f)
    #Perturbations
    du = -ygrid/R*Omega
    dv = xgrid/R*Omega
    dT = -(gamma-1)/2*Omega**2
    #Initial conditions
    state = np.zeros(xgrid.shape+(4,))
    #THESE ARE NONDIMENSIONAL - NEED INTERNET
    state[:,:,0] = (1+dT)**(1/(gamma-1))    #Initial density
    state[:,:,1] = M_inf*np.cos(alpha)+du   #Initial u
    state[:,:,2] = M_inf*np.cos(alpha)+du   #Initial v
    state[:,:,3] = 1/gamma*(1+dT)**(gamma/(1-gamma))   #Initial pressure
    print(state)







if __name__ == "__main__":
    #HighOrder Workshop Fast
    ICs = Shu()
    args = ICs.get()
    vortex(args)

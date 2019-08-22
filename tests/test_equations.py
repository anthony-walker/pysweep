#Programmer: Anthony Walker
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
from analytical import *
from equations import *
from sweep import *
import numpy as np

def test_python_euler():
    """Use this function to test the python version of the euler code."""
    #Properties
    gamma = 1.4

    #Analytical properties
    cvics = vics()
    cvics.Shu(gamma)
    initial_args = cvics.get_args()
    X = cvics.L
    Y = cvics.L
    #Dimensions and steps
    npx = 16
    npy = 16
    dx = X/npx
    dy = Y/npy

    #Time testing arguments
    t0 = 0
    t_b = 0.2
    dt = 0.01
    targs = (t0,t_b,dt)
    #Analytical Array
    analyt_vortex = vortex(cvics,X,Y,npx,npy,times=(0,0.1,0.2))
    analyt_vortex = np.swapaxes(analyt_vortex,0,2)
    analyt_vortex = np.swapaxes(analyt_vortex,1,3)
    #Numerical Array
    time_steps = int((t_b-t0)/dt)
    num_vortex = np.zeros((time_steps,)+analyt_vortex.shape[1:])
    num_vortex[0,:,:,:] = analyt_vortex[0,:,:,:]

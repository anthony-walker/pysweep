#Programmer: Anthony Walker
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
from analytical import *
from equations import *
from sweep import *
from decomp import *
import numpy as np

dtdx = 0
dtdy = 0
gamma = 0


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
    npx = 96
    npy = 96
    dx = (2*X)/npx
    dy = (2*Y)/npy
    #Time testing arguments
    t0 = 0
    t_b = 0.1
    dt = 0.01
    targs = (t0,t_b,dt)
    nta = 2
    jump = int(t_b/(dt*(nta-1)))
    #Analytical Array
    comp_times = np.linspace(t0,t_b,nta)    #Comparison times
    analyt_vortex = vortex(cvics,X,Y,npx,npy,times=comp_times)
    flux_vortex = convert_to_flux(analyt_vortex[:,:,:,:],gamma)
    #Numerical Array
    time_steps = int((t_b-t0)/dt)   #number of time steps
    avs = analyt_vortex.shape
    ops = 2
    #Create flux vortex array
    num_vortex = np.zeros((time_steps+1,avs[1],avs[2]+2*ops,avs[3]+2*ops))
    num_vortex[0,:,ops:-ops,ops:-ops] = flux_vortex[0,:,:,:]
    num_vortex[0,:,:ops,:ops] = num_vortex[0,:,-2*ops:-ops,-2*ops:-ops]
    num_vortex[0,:,-ops:,-ops:] = num_vortex[0,:,ops:2*ops,ops:2*ops]
    #Testing
    #Setting globals
    set_cpu_globals((dx,dy,dt,gamma))
    #Get source module
    source_mod = build_cpu_source("./src/equations/euler.py")
    #Create iidx sets
    iidx = tuple()
    for i in range(npx):
        for j in range(npy):
            iidx+=(i+ops,j+ops),

    # Solve sets
    for ts in range(time_steps):
        num_vortex = step(num_vortex,iidx,ts)
        #Periodic bc
        num_vortex[ts+1,:,:ops,:ops] = num_vortex[ts+1,:,-2*ops:-ops,-2*ops:-ops]
        num_vortex[ts+1,:,-ops:,-ops:] = num_vortex[ts+1,:,ops:2*ops,ops:2*ops]

    #Testing if close
    for ai,i in enumerate(range(0,time_steps+1,jump)):
        res = np.isclose(flux_vortex[ai,:,:,:],num_vortex[i,:,ops:-ops,ops:-ops])
        assert res.all() == True

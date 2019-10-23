#Programmer: Anthony Walker
#This file contains a numerical solution to the 2D unsteady heat diffusion
# equation with RK2

import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    print(e)
#----------------------------------Globals-------------------------------------#
alpha = 0
dtdx2 = 0
dtdy2 = 0

def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    vs = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        dT = d2Tdxy(state,(ts,vs,idx,idy))
        if (gts+1)%2==0:   #Corrector step
            state[ts+1,vs,idx,idy] = state[ts-1,vs,idx,idy]+dT
        else: #Predictor step
            state[ts+1,vs,idx,idy] = state[ts,vs,idx,idy]+half*dT
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy,alp = args
    #Scheme stability
    Fox = alp*dt/(dx*dx)
    Foy = alp*dt/(dy*dy)
    criteria = 0.25
    assert Fox <= criteria
    assert Foy <= criteria
    #Setting Globals
    if gpu:
        keys = "DT","DX","DY","ALPHA","DTDX2","DTDY2"
        nargs = args[2:]+(dt/(dx*dx),dt/(dy*dy))
        fc = lambda x:np.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        global dtdx2
        dtdx2 = dt/(dx*dx)
        global dtdy2
        dtdy2 = dt/(dy*dy)
        global alpha
        alpha = alp

def d2Tdxy(state,idx):
    #Five point finite volume method
    #Creating indices from given point (idx)
    ops = 1 #Stencil size on each side
    id1,id2,id3,id4 = idx
    #Finding spatial derivatives
    dT = alpha*dtdx2*(state[id1,id2,id3+ops,id4]+state[id1,id2,id3-ops,id4]-2*state[id1,id2,id3,id4])
    dT += alpha*dtdy2*(state[id1,id2,id3,id4+ops]+state[id1,id2,id3,id4-ops]-2*state[id1,id2,id3,id4])
    return dT

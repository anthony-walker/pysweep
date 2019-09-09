#Programmer: Anthony Walker
#This file contains a numerical solution to the 2D unsteady heat diffusion
# equation with RK2

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
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
        dTx,dTy = d2Tdxy(state,(ts,vs,idx,idy))
        if (gts+1)%2==0:   #Corrector step
            state[ts+1,vs,idx,idy] = state[ts-1,vs,idx,idy]+half*dTx+half*dTy
        else: #Predictor step
            state[ts+1,vs,idx,idy] = state[ts,vs,idx,idy]+half*dTx+half*dTy
    return state

def set_globals(gpu,source_mod,*args):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy,alp = args
    if gpu:
        keys = "DT","DX","DY","DTDX2","DTDY2","ALPHA"
        nargs = args[2:]+(dt/(dx*dx),dt/(dy*dy),alp)
        fc = lambda x:np.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        global dtdx2
        dtdx = dt/(dx*dx)
        global dtdy2
        dtdy = dt/(dy*dy)
        global alpha
        alpha = alp

def d2Tdxy(state,idx):
    #Five point finite volume method
    #Creating indices from given point (idx)
    ops = 1 #Stencil size on each side
    id1,id2,id3,id4 = idx
    idxx=(id1,id2,slice(id3-ops,id3+ops+1,1),id4)
    idxx=(id1,id2,id3,slice(id4-ops,id4+ops+1,1))
    #Finding spatial derivatives
    dTx = alpha*dtdx2*(state[id1,id2,id3+1,id4]+state[id1,id2,id3-1,id4]+2*state[id1,id2,id3,id4])
    dTy = alpha*dtdx2*(state[id1,id2,id3,id4+1]+state[id1,id2,id3,id4-1]+2*state[id1,id2,id3,id4])
    return dTx, dTy

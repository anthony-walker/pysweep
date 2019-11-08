#Programmer: Anthony Walker
import sys, os
import numpy as np
import warnings
import itertools
warnings.simplefilter('ignore')
fp = os.path.abspath(__file__)
path = os.path.dirname(fp)
sys.path.insert(0, path[:-5])
import distributed.sweep.ccore.source as source
import distributed.sweep.ccore.printer as printer
epath = os.path.join(path[:-5],'equations')

def test_steps():
    """Use this function to test several steps for 1 point"""
    #Sod Shock BC's
    t0 = 0
    tf = .1
    dt = 0.01
    X = Y = 1
    gamma = 1.4
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    nx = ny = 20
    nxh = int(ny/2)
    nv = 4
    ops = 2
    dx = X/nx
    dy = Y/ny

    #2D array
    test_shape = (int((tf-t0)/dt)+1,nv,nx,ny)
    arr2D = np.zeros(test_shape)
    for i in range(nx):
        for j in range(nxh):
            arr2D[0,:,i,j] = leftBC
            arr2D[0,:,i,j+nxh] = rightBC
    idx2D = list(itertools.product((int(nx/2),),np.arange(ops,arr2D.shape[3]-ops)))
    #Source Modules
    source_mod_2D = source.build_cpu_source(os.path.join(epath,'euler.py'))
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = source.build_cpu_source(os.path.join(epath,'euler1D.py'))
    steps = test_shape[0]
    for i in range(steps):
        arr2D = source_mod_2D.step(arr2D,idx2D,i,i)
        #Update boundary - periodic
        arr2D[i+1,:,:,:ops] = arr2D[i,:,:,-2*ops:-ops]
        arr2D[i+1,:,:,-ops:] = arr2D[i,:,:,ops:2*ops]
        arr2D[i+1,:,:ops,:] = arr2D[i,:,-2*ops:-ops,:]
        arr2D[i+1,:,-ops:,:] = arr2D[i,:,ops:2*ops,:]
        # print("--------------------------------------------")
        printer.pm(arr2D,i,ps="%0.4f")



if __name__ == "__main__":
    test_steps()

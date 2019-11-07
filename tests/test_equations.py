#Programmer: Anthony Walker
import sys, os
import numpy as np
import warnings
warnings.simplefilter('ignore')
fp = os.path.abspath(__file__)
path = os.path.dirname(fp)
sys.path.insert(0, path[:-5])
import distributed.sweep.ccore.source as source
epath = os.path.join(path[:-5],'equations')

def test_steps():
    """Use this function to test several steps for 1 point"""
    #Sod Shock BC's
    t0 = 0
    tf = .1
    dt = 0.01
    dx = 0.1
    dy = 0.1
    gamma = 1.4
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    nx = ny = 10
    nxh = int(ny/2)
    nv = 4
    ops = 2
    #2D array
    test_shape = (int((tf-t0)/dt),nv,nx,ny)
    arr2D = np.zeros(test_shape)
    for i in range(nx):
        for j in range(nxh):
            arr2D[0,:,i,j] = leftBC
            arr2D[0,:,i,j+nxh] = rightBC

    print(arr2D[0])
    #Source Modules
    source_mod_2D = source.build_cpu_source(os.path.join(epath,'euler.py'))
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = source.build_cpu_source(os.path.join(epath,'euler1D.py'))




if __name__ == "__main__":
    test_steps()

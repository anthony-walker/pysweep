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

def test_flux():
    """Use this function to test the python version of the euler code.
    This test uses a formerly validate 1D code and computes the fluxes in each direction
    to test whether the 2D fluxes are correctly computing
    """
    #Sod Shock BC's
    t0 = 0
    tf = 1
    dt = 0.01
    dx = 0.1
    dy = 0.1
    gamma = 1.4
    leftBC = (1000,0,0,2500)
    rightBC = (0.125,0,0,.25)
    test_shape = (2,4,5,5)
    num_test = np.zeros((2,4,5,5))
    for i in range(3):
        for j in range(3):
            num_test[0,:,i,j]=leftBC[:]
    for i in range(2,5):
        for j in range(2,5):
            num_test[0,:,i,j]=rightBC[:]
    xtest = np.zeros(test_shape)
    ytest = np.zeros(test_shape)
    for i in range(2):
        xtest[0,:,i,2]=leftBC
        xtest[0,:,i+1,2]=rightBC
        xtest[0,:,i+3,2]=rightBC
        ytest[0,:,2,i]=leftBC
        ytest[0,:,2,i+1]=rightBC
        ytest[0,:,2,i+3]=rightBC
    Qx = np.zeros((5,3))
    Qx[:,0] = num_test[0,0,:,2]
    Qx[:,1] = num_test[0,1,:,2]
    Qx[:,2] = num_test[0,3,:,2]
    P = np.zeros(5)
    P[:] = [1000,1000,0.1,0.1,0.1]
    #Get source module
    source_mod_2D = source.build_cpu_source(os.path.join(epath,'euler.py'))
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = source.build_cpu_source(os.path.join(epath,'euler1D.py'))
    iidx = (0,slice(0,4,1),2,2)
    #Testing Flux
    # print(num_test)
    dfdx,dn = source_mod_2D.dfdxy(xtest,iidx)
    # dn,dfdy = source_mod_2D.dfdxy(ytest,iidx)
    # dfdx = np.delete(dfdx,2)
    # dfdy = np.delete(dfdy,1)
    # df = source_mod_1D.fv5p(Qx,P)[2]
    # assert np.isclose(df.all(),dfdx.all())
    # assert np.isclose(df.all(),dfdy.all())
    # print(df)
    # print(dfdy)
    # print(dfdx)
if __name__ == "__main__":
    test_flux()

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
from analytical import vortex
epath = os.path.join(path[:-5],'equations')
#Cuda imports
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass

def test_steps():
    """Use this function to test several steps for 1 point"""
    #Sod Shock BC's
    t0 = 0
    tf = 1
    dt = 0.01
    X = Y = 1
    gamma = 1.4
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    # leftBC = (1,1,0,0) #P, rho, u,v
    # rightBC = (0.1,0.125,0,0)
    nx = ny = 24
    BS = 12
    nxh = int(ny/2)
    nv = 4
    OPS =ops = 2
    TSO = 2
    dx = X/nx
    dy = Y/ny
    MPSS = 2
    MOSS = 4

    #Analytical properties
    # cvics = vortex.vics()
    # cvics.Shu(gamma)
    # arr2D = vortex.steady_vortex(cvics,nx,ny)
    # arr2D = cvics.flux
    test_shape = (int((tf-t0)/dt)+1,nv,nx,ny)
    block_shape=test_shape
    arr2D = np.zeros(test_shape,dtype=np.float32)

    XY = True
    idxy = 2 if XY else 1

    for i in range(nx):
        # for j in range(0,nx-i,1):
        #     arr2D[0,:,i,j] = leftBC
        # for j in range(-i,0,1):
        #     arr2D[0,:,i,j] = rightBC
        if XY:
            # Vary in y
            for j in range(nxh):
                arr2D[0,:,i,j] = leftBC
                arr2D[0,:,i,j+nxh] = rightBC
        else:
            #Vary in x
            for j in range(nxh):
                arr2D[0,:,j,i] = leftBC
                arr2D[0,:,j+nxh,i] = rightBC
    arr2D[1]=arr2D[0]
    # if XY:
    #     arr2D[1]=arr2D[0]
    #     arr2D[1,0,:,nxh-1]=0.97480232
    #     arr2D[1,0,:,nxh]=0.15019771
    #     arr2D[1,2,:,nxh-1]=0.02250000
    #     arr2D[1,2,:,nxh]=0.02250000
    #     arr2D[1,3,:,nxh-1]=2.43520594
    #     arr2D[1,3,:,nxh]=0.31479412
    # else:
    #     arr2D[1]=arr2D[0]
    #     arr2D[1,0,nxh-1,:]=0.97480232
    #     arr2D[1,0,nxh,:]=0.15019771
    #     arr2D[1,2,nxh-1,:]=0.02250000
    #     arr2D[1,2,nxh,:]=0.02250000
    #     arr2D[1,3,nxh-1,:]=2.43520594
    #     arr2D[1,3,nxh,:]=0.31479412
    # arr2D = np.ones()
    # arr2D = vortex.convert_to_flux(arr2D,gamma)
    #GPU_stuff
    GRD = (int((block_shape[2])/BS),int((block_shape[3])/BS))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SGIDS = (nx)*(ny)
    STS = SGIDS*NV #Shared time shift
    VARS =  block_shape[2]*(block_shape[3])
    TIMES = VARS*NV

    const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})

    #Source Modules
    SM = source.build_gpu_source(os.path.join(epath,'euler.h'))
    source.swept_constant_copy(SM,const_dict)
    arr_gpu = cuda.mem_alloc(arr2D.nbytes)

    # printer.pm(arr2D[:,:,:,:],1,iv=0,ps="%0.4f")

    for i in range(0,4):
        cuda.memcpy_htod(arr_gpu,arr2D)
        SM.get_function('test_step')(arr_gpu,np.int32(0),np.int32(i),grid=GRD,block=(BS,BS,1))
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(arr2D,arr_gpu)
        if XY:
            for j in range(nx):
                for k in range(ops):
                    arr2D[1,:,j,k] = leftBC
                    arr2D[1,:,j,-(k+1)] = rightBC

            arr2D[1,:,:ops,:] = arr2D[1,:,-2*ops:-ops,:]
            arr2D[1,:,-ops:,:] = arr2D[1,:,ops:2*ops,:]
        else:
            for k in range(nx):
                for j in range(ops):
                    arr2D[1,:,j,k] = leftBC
                    arr2D[1,:,-(j+1),k] = rightBC
            arr2D[1,:,:,:ops] = arr2D[1,:,:,-2*ops:-ops]
            arr2D[1,:,:,-ops:] = arr2D[1,:,:,ops:2*ops]

        if (i+1)%2==0:
            arr2D[0] = arr2D[1]
        if XY:
            printer.pm(arr2D[:,:,nxh:nxh+1,:],1,iv=0,ps="%0.5f")
            printer.pm(arr2D[:,:,nxh:nxh+1,:],1,iv=idxy,ps="%0.5f")
            printer.pm(arr2D[:,:,nxh:nxh+1,:],1,iv=3,ps="%0.5f")
        else:

            printer.pm(arr2D[:,:,:,nxh:nxh+1],1,iv=0,ps="%0.5f")
            printer.pm(arr2D[:,:,:,nxh:nxh+1],1,iv=idxy,ps="%0.5f")
            printer.pm(arr2D[:,:,:,nxh:nxh+1],1,iv=3,ps="%0.5f")
        print('---------------------'+str(i)+'-----------------------')
        input()


if __name__ == "__main__":
    test_steps()

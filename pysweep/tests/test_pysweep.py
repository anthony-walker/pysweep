#Programmer: Anthony Walker
#Use the functions in this file to test the decomposition code.
import sys, os, h5py
sys.path.insert(0, './pysweep')
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sweep.ncore import block, mplambda

def map_test_fcn(args):
    blockc,tn = args
    blockc[1,:,:,:] = blockc[0,:,:,:]
    return blockc

def test_block_management(args=None):
    x = 20
    BS = (int(x/2),int(x/2),1)
    ops = 2
    arr = np.zeros((10,4,x+2*ops,x+2*ops))
    CRS = block.create_blocks_list(arr.shape,BS,ops)
    arr[0,:,2:12,2:12] = 1
    arr[0,:,2:12,12:22] = 2
    arr[0,:,12:22,2:12] = 3
    arr[0,:,12:22,12:22] = 4
    blocks = list()
    for local_region in CRS:
        blockc = np.copy(arr[local_region])
        blocks.append(blockc)
    cpu_fcn = mplambda.sweep_lambda((map_test_fcn,1))
    blocks = list(map(cpu_fcn,blocks))
    arr = block.rebuild_blocks(arr,blocks,CRS,ops)
    assert ((arr[1,:,:,:]-arr[0,:,:,:])==0).all()

def test_sweep_write(args=None):
    estr = "mpiexec -n 8 python ./pysweep/pst.py stest "
    estr += "-b 10 -o 2 --tso 2 -a 0.5 -g \"./pysweep/equations/eqt.h\" -c \"./pysweep/equations/eqt.py\" "
    estr += "--hdf5 \"./stest\" -nx 40 -ny 40"
    os.system(estr)
    test_file = "./stest.hdf5"
    hdf5_file = h5py.File(test_file, 'r')
    hdf5_data_set = hdf5_file['data']
    for i in range(1,len(hdf5_data_set[:,0,:,:])):
        if not (hdf5_data_set[i,0,:,:]==0).all():
            assert (hdf5_data_set[i,0,:,:]-hdf5_data_set[i-1,0,:,:]==2).all()
    os.system("rm "+test_file)

def plot_step(data,t,i,npx,X):
    npy=npx
    Y = X
    xpts = np.linspace(-X,X,npx,dtype=np.float64)
    ypts = np.linspace(-Y,Y,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
    fig, ax =plt.subplots()
    ax.set_ylim(-Y, Y)
    ax.set_xlim(-X, X)
    ax.set_title("Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(-1.2,1.2,100))
    ax.contourf(xgrid,ygrid,data[t,i,:,:],levels=40,cmap=cm.inferno)
    prim_path = "./pysweep/tests/data/imgs/"
    pn = "fig0"
    png =".png"
    ct = 1
    while os.path.isfile(prim_path+pn+png):
        pn = pn[:3]+str(ct)
        ct+=1
    plt.savefig(prim_path+pn+png)


def test_eqt2(args=(1,40,0,10,5,10,4)):
    """Use this function to troubleshoot the swept rule"""
    swept_file = "\"./pysweep/tests/data/swept_eqt2\""
    sfp = "./pysweep/tests/data/swept_eqt2.hdf5"
    os.system("rm "+sfp)
    tf,npx,aff,X,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = 0.01
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "ccde mpiexec -n "+str(nps)+" python ./pysweep/pst.py stest2 "
        estr += "-b "+str(blks)+" -o 2 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/eqt2.h\" -c \"./pysweep/equations/eqt2.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str
        os.system(estr)

def test_sweep_vortex(args=(1,0.01,40,0,10,10,4)):
    swept_file = "\"./pysweep/tests/data/swept_vortex\""
    sfp = "./pysweep/tests/data/swept_vortex.hdf5"
    os.system("rm "+sfp)
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python ./pysweep/pst.py swept_vortex "
        estr += "-b 10 -o 2 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/euler.h\" -c \"./pysweep/equations/euler.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str
        os.system(estr)

def test_sweep_hde(args=(8,40,1,10,0.24,5,10,4)):
    savepath = "./swept_hde_plot"
    swept_file = "\"./pysweep/tests/data/swept_hde\""
    sfp = "./pysweep/tests/data/swept_hde.hdf5"
    afp = "./pysweep/tests/data/analyt_hde0.hdf5"
    analyt_file = "\"./pysweep/tests/data/analyt_hde\""
    os.system("rm "+sfp)

    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "ccde mpiexec -n "+str(nps)+" python ./pysweep/pst.py swept_hde "
        estr += "-b "+str(blks)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/hde.h\" -c \"./pysweep/equations/hde.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
        os.system(estr)

def test_dsweep_hde(args=(8, 120, 0.5, 10, 0.24, 5, 12, 3)):
    savepath = "./swept_hde_plot"
    swept_file = "\"./pysweep/tests/data/dswept_hde\""
    sfp = "./pysweep/tests/data/dswept_hde.hdf5"
    afp = "./pysweep/tests/data/analyt_hde0.hdf5"
    analyt_file = "\"./pysweep/tests/data/analyt_hde\""
    os.system("rm "+sfp)

    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(nps)+"--hostfile=sub-nodes python ./pysweep/pst.py DSHDE "
        estr += "-b "+str(blks)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/hde.h\" -c \"./pysweep/equations/hde.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
        os.system(estr)

if __name__ == "__main__":
    test_dsweep_hde()
    # test_sweep_vortex()
    # test_sweep_hde()
    # test_eqt2()
    # test_sweep_write()
    # test_block_management()

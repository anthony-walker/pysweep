import os, sys, h5py
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
fp = os.path.abspath(__file__)
path = os.path.dirname(fp)
sys.path.insert(0, path[:-5])
import distributed.sweep.ccore.source as source
import distributed.sweep.ccore.printer as printer
import analytical.vortex as vortex
import analytical.sodShock as sodshock
import equations.test_euler as teuler
import equations.euler as euler

from multiprocessing import Pool
epath = os.path.join(path[:-5],'equations')
dpath = os.path.join(path,'data')
sfs="%0.5f"
# import warnings
# warnings.simplefilter('ignore')

def test_swept_block(args=(0,0.4,0.001,24,2.4,1,1,12),remove_file=False,generate_fig=False,hoststr=""):
    """This funciton is intended to validate the solvers in 2D."""
    sfn = os.path.join(dpath,"dist_swept_simple")
    swept_file = "\""+sfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    t0,tf,dt,npx,X,nps,aff,blks = args
    npy=npx
    dx = X/(npx-1)
    Y=X
    gamma = 1.4
    ops=2
    nt = int((tf-t0)/dt) #number of time steps
    #Exe strigns
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    print("Executing Swept Block")
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" swept_simple --distributed \'true\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(gamma)
    os.system(estr)

    #Testing against 1D
    propidx = 0
    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    swept_arr = swept_hdf5['data'][:,:,:,:]
    for t,arr in enumerate(swept_arr):
        ct = 0
        for idz in range(arr.shape[0]):
            for idx in range(arr.shape[1]):
                for idy in range(arr.shape[2]):
                    assert arr[idz,idx,idy]==ct+t#:
                    ct+=1
    print("Blocksize test completed successfully.")
    swept_hdf5.close()
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)

def test_standard_block(args=(0,0.1,0.01,24,2.4,1,0,12),remove_file=False,generate_fig=False,hoststr=""):
    """This funciton is intended to validate the solvers in 2D."""
    sfn = os.path.join(dpath,"dist_stand_simple")
    swept_file = "\""+sfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    t0,tf,dt,npx,X,nps,aff,blks = args
    npy=npx
    dx = X/(npx-1)
    Y=X
    gamma = 1.4
    ops=2
    nt = int((tf-t0)/dt) #number of time steps
    #Exe strigns
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    print("Executing Standard Block")
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" standard_simple --distributed \'true\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(gamma)
    os.system(estr)

    #Testing against 1D
    propidx = 0
    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    swept_arr = swept_hdf5['data'][:,:,:,:]
    for t,arr in enumerate(swept_arr):
        ct = 0
        for idz in range(arr.shape[0]):
            for idx in range(arr.shape[1]):
                for idy in range(arr.shape[2]):
                    assert arr[idz,idx,idy]==ct+t#:
                    ct+=1
    print("Blocksize test completed successfully.")
    swept_hdf5.close()
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)

def test_cpu_step(args=(0,1,0.01,12,1.2)):
    """Use this function to test the cpu step functions"""
    t0,tf,dt,npx,X = args
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    npy=npx
    dx = X/(npx-1)
    Y=X
    gamma = 1.4
    ops=2
    nt = int((tf-t0)/dt) #number of time steps
    arr2D = np.zeros((2*nt,4,npx+2*ops,npy+2*ops))
    iidx = [(idx+ops,idy+ops)for idx,idy in np.ndindex((npx,npy))]
    nyh = int(npy/2)
    nxh = int(npx/2)
    nxq = int(nxh/2)
    nyq = int(nyh/2)
    #---------------------------------1D SHOCK-----------------------------------#
    lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
    rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
    teuler.set_globals(gamma,dt,dx)
    shock1D = np.zeros((2*nt,3,npx+2*ops))
    halfx = int(npx/2)
    halfy = int(npy/2)
    for x in range(halfx):
        shock1D[0,:,x+ops] = lbc1
        shock1D[0,:,x+halfx+ops] = rbc1
    shock1D[0,:,:ops] = shock1D[0,:,-2*ops:-ops]
    shock1D[0,:,-ops:] = shock1D[0,:,ops:2*ops]
    shock1D[1,:,ops:-ops] = teuler.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
    shock1D[1,:,:ops] = shock1D[1,:,-2*ops:-ops]
    shock1D[1,:,-ops:] = shock1D[1,:,ops:2*ops]
    for i in range(1,2*nt-1):
        #1D
        id1 = i-1 if (i+1)%2==0 else i
        shock1D[i+1,:,ops:-ops] = teuler.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
        shock1D[i+1,:,:ops] = shock1D[i+1,:,-2*ops:-ops]
        shock1D[i+1,:,-ops:] = shock1D[i+1,:,ops:2*ops]

    #---------------------------------Globals-----------------------------------------------#
    euler.set_globals(False,t0,tf,dt,dx,dx,gamma)
    #---------------------------------2D SHOCK X -Direction-----------------------------------#
    print("---------------------------------2D SHOCK X -Direction-----------------------------------")
    arr2Dx = np.zeros((nt,4,npx,npy))
    for i in range(nxh):
        for j in range(npy):
            arr2D[0,:,i+ops,j+ops] = leftBC
            arr2D[0,:,i+nxh+ops,j+ops] = rightBC
    arr2D[0,:,:ops,:] = arr2D[0,:,-2*ops:-ops,:]
    arr2D[0,:,-ops:,:] = arr2D[0,:,ops:ops+ops,:]
    arr2D[0,:,:,:ops] = arr2D[0,:,:,-2*ops:-ops]
    arr2D[0,:,:,-ops:] = arr2D[0,:,:,ops:ops+ops]

    #First step
    i=0
    euler.step(arr2D,iidx,i,i)
    arr2D[i+1,:,:ops,:] = arr2D[i+1,:,-2*ops:-ops,:]
    arr2D[i+1,:,-ops:,:] = arr2D[i+1,:,ops:ops+ops,:]
    arr2D[i+1,:,:,:ops] = arr2D[i+1,:,:,-2*ops:-ops]
    arr2D[i+1,:,:,-ops:] = arr2D[i+1,:,:,ops:ops+ops]
    #Iteration
    for i in range(1,2*nt-1):
        #2D
        euler.step(arr2D,iidx,i,i)
        arr2D[i+1,:,:ops,:] = arr2D[i+1,:,-2*ops:-ops,:]
        arr2D[i+1,:,-ops:,:] = arr2D[i+1,:,ops:ops+ops,:]
        arr2D[i+1,:,:,:ops] = arr2D[i+1,:,:,-2*ops:-ops]
        arr2D[i+1,:,:,-ops:] = arr2D[i+1,:,:,ops:ops+ops]

    for i in range(1,2*nt-1):
        for j,p in enumerate((0,1,3)):
            for cy in range(npy):
                tarr = arr2D[i+1,p,:,cy]-shock1D[i+1,j,:]
                if np.where(tarr > 1e-14)[0].size > 0:
                    print('-----------------------------'+str(i)+'--------------------------------')
                    sys.stdout.write('[')
                    for item in tarr:
                        sys.stdout.write("%0.2e"%item+", ")
                    sys.stdout.write(']\n')
                    sys.stdout.write('[')
                    for item in arr2D[i+1,p,:,cy]:
                        sys.stdout.write(sfs%item+", ")
                    sys.stdout.write(']\n')
                    sys.stdout.write('[')
                    for item in shock1D[i+1,j,:]:
                        sys.stdout.write(sfs%item+", ")
                    sys.stdout.write(']\n')
                    assert False, "Failed comparison"
        if i == 2*nt-2:
            print("successfully completed x direction flux.")
    #---------------------------------2D SHOCK Y -Direction-----------------------------------#
    print("---------------------------------2D SHOCK Y -Direction-----------------------------------")
    arr2D = np.zeros((2*nt,4,npx+2*ops,npy+2*ops))
    for i in range(nyh):
        for j in range(npx):
            arr2D[0,:,j+ops,i+ops] = leftBC
            arr2D[0,:,j+ops,i+nyh+ops] = rightBC
    arr2D[0,:,:,:ops] = arr2D[0,:,:,-2*ops:-ops]
    arr2D[0,:,:,-ops:] = arr2D[0,:,:,ops:ops+ops]
    arr2D[0,:,:ops,:] = arr2D[0,:,-2*ops:-ops,:]
    arr2D[0,:,-ops:,:] = arr2D[0,:,ops:ops+ops,:]

    #Indexing for testing if necessary
    pidx = 2
    pidx2 = pidx if pidx < 1 else pidx+1

    for i in range(0,2*nt-1):
        euler.step(arr2D,iidx,i,i)
        arr2D[i+1,:,:,:ops] = arr2D[i+1,:,:,-2*ops:-ops]
        arr2D[i+1,:,:,-ops:] = arr2D[i+1,:,:,ops:ops+ops]
        arr2D[i+1,:,:ops,:] = arr2D[i+1,:,-2*ops:-ops,:]
        arr2D[i+1,:,-ops:,:] = arr2D[i+1,:,ops:ops+ops,:]


    for i in range(1,2*nt-1):
        for j,p in enumerate((0,2,3)):
            for cx in range(npx):
                tarr = arr2D[i+1,p,cx,:]-shock1D[i+1,j,:]
                if np.where(tarr > 1e-14)[0].size > 0:
                    print('-----------------------------'+str(i)+'--------------------------------')
                    sys.stdout.write('[')
                    for item in tarr:
                        sys.stdout.write("%0.2e"%item+", ")
                    sys.stdout.write(']\n')
                    sys.stdout.write('[')
                    for item in arr2D[i+1,p,cx,:]:
                        sys.stdout.write(sfs%item+", ")
                    sys.stdout.write(']\n')
                    sys.stdout.write('[')
                    for item in shock1D[i+1,j,:]:
                        sys.stdout.write(sfs%item+", ")
                    sys.stdout.write(']\n')
                    assert False, "Failed comparison"
        if i == 2*nt-2:
            print("successfully completed y direction flux.")

def test_swept_shock(args=(0,0.5,0.01,24,2.4,1,1,12),remove_file=True,generate_fig=False,assertion=True,xy = True):
    """This funciton is intended to validate the solvers in 2D."""
    sfn = os.path.join(dpath,"dist_swept_shock")
    swept_file = "\""+sfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    t0,tf,dt,npx,X,nps,aff,blks = args
    npy=npx
    dx = X/(npx-1)
    Y=X
    gamma = 1.4
    ops=2
    nt = int((tf-t0)/dt) #number of time steps
    orient = 0 if xy else 1
    #Exe strigns
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    print("Executing Swept Shock")
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_shock --distributed \'true\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" -o "+str(orient)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(gamma)
    os.system(estr)

    #1D shock
    leftBC = (1,0,0,2.5)
    lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
    rightBC = (0.125,0,0,.25)
    rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
    teuler.set_globals(gamma,dt,dx)
    shock1D = np.zeros((2*nt,3,npx+2*ops))
    halfx = int(npx/2)
    halfy = int(npy/2)
    for x in range(halfx):
        shock1D[0,:,x+ops] = lbc1
        shock1D[0,:,x+halfx+ops] = rbc1
    shock1D[0,:,:ops] = shock1D[0,:,-2*ops:-ops]
    shock1D[0,:,-ops:] = shock1D[0,:,ops:2*ops]
    shock1D[1,:,ops:-ops] = teuler.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
    shock1D[1,:,:ops] = shock1D[1,:,-2*ops:-ops]
    shock1D[1,:,-ops:] = shock1D[1,:,ops:2*ops]
    for i in range(1,2*nt-1):
        #1D
        id1 = i-1 if (i+1)%2==0 else i
        shock1D[i+1,:,ops:-ops] = teuler.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
        shock1D[i+1,:,:ops] = shock1D[i+1,:,-2*ops:-ops]
        shock1D[i+1,:,-ops:] = shock1D[i+1,:,ops:2*ops]
    shock1Df = np.zeros((nt,3,npx))
    for i,k in enumerate(range(0,2*nt,2)):
        shock1Df[i,:,:] = shock1D[k,:,ops:-ops]
    #Testing against 1D
    propidx = 1
    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    swept_arr = swept_hdf5['data'][:,:,:,:]

    for i in range(min(len(swept_arr),len(shock1Df))):
        #2D
        if xy:
            cerr = np.asarray([np.amax(swept_arr[i,propidx,:,y]-shock1Df[i,propidx,:]) for y in range(npy)])
        else:
            cerr = np.asarray([np.amax(swept_arr[i,propidx,x,:]-shock1Df[i,propidx,:]) for x in range(npx)])
        if assertion:
            assert np.where(cerr > 1e-10)[0].size < 1
        else:
            print(np.amax(cerr))
    print("Shock test completed successfully.")
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)

def test_standard_shock(args=(0,0.5,0.01,24,2.4,1,1,12),remove_file=True,generate_fig=False,assertion=True,xy=True):
    """This funciton is intended to validate the solvers in 2D."""
    sfn = os.path.join(dpath,"dist_stand_shock")
    swept_file = "\""+sfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    t0,tf,dt,npx,X,nps,aff,blks = args
    npy=npx
    dx = X/(npx-1)
    Y=X
    gamma = 1.4
    ops=2
    orient = 0 if xy else 1
    nt = int((tf-t0)/dt) #number of time steps
    #Exe strigns
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    print("Executing Standard Shock")
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_shock --distributed \'true\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" -o "+str(orient)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(gamma)
    os.system(estr)

    #1D shock
    leftBC = (1,0,0,2.5)
    lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
    rightBC = (0.125,0,0,.25)
    rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
    teuler.set_globals(gamma,dt,dx)
    shock1D = np.zeros((2*nt,3,npx+2*ops))
    halfx = int(npx/2)
    halfy = int(npy/2)
    for x in range(halfx):
        shock1D[0,:,x+ops] = lbc1
        shock1D[0,:,x+halfx+ops] = rbc1
    shock1D[0,:,:ops] = shock1D[0,:,-2*ops:-ops]
    shock1D[0,:,-ops:] = shock1D[0,:,ops:2*ops]
    shock1D[1,:,ops:-ops] = teuler.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
    shock1D[1,:,:ops] = shock1D[1,:,-2*ops:-ops]
    shock1D[1,:,-ops:] = shock1D[1,:,ops:2*ops]
    for i in range(1,2*nt-1):
        #1D
        id1 = i-1 if (i+1)%2==0 else i
        shock1D[i+1,:,ops:-ops] = teuler.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
        shock1D[i+1,:,:ops] = shock1D[i+1,:,-2*ops:-ops]
        shock1D[i+1,:,-ops:] = shock1D[i+1,:,ops:2*ops]
    shock1Df = np.zeros((nt,3,npx))
    for i,k in enumerate(range(0,2*nt,2)):
        shock1Df[i,:,:] = shock1D[k,:,ops:-ops]
    #Testing against 1D
    propidx = 0
    stand_hdf5 = h5py.File(sfn+".hdf5", 'r')
    stand_arr = stand_hdf5['data'][:,:,:,:]

    for i in range(min(len(stand_arr),len(shock1Df))):
        #2D
        if xy:
            cerr = np.asarray([np.amax(stand_arr[i,propidx,:,y]-shock1Df[i,propidx,:]) for y in range(npy)])
        else:
            cerr = np.asarray([np.amax(stand_arr[i,propidx,x,:]-shock1Df[i,propidx,:]) for x in range(npx)])
        if assertion:
            assert np.where(cerr > 1e-10)[0].size < 1
        else:
            print(np.amax(cerr))
    print("Shock test completed successfully.")
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)

def test_validate_vortex(args=(0,1,0.001,48,10,12,0,1),remove_file=True,hoststr="",efn = "swept_err.hdf5"):
    """Use this funciton to validate the swept solver with a 2D euler vortex"""

    #--------------------------------_Swept Vortex-------------------------------------#
    swept_file = "\""+os.path.join(dpath,"dist_swept_vortex")+"\""
    sfn = os.path.join(dpath,"dist_swept_vortex.hdf5")
    sfp = "\""+sfn+"\""
    t0,tf,dt,npx,X,blks,aff,nps = args
    npy=npx
    Y=X
    gamma = 1.4
    times = np.arange(t0,tf+dt,dt)
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "

    if not os.path.isfile(sfn):
    #Create data using solver
        print("Executing Swept Unsteady Vortex")
        estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" unsteady_swept_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + swept_file + pts +time_str
        os.system(estr)


    decomp_file = "\""+os.path.join(dpath,"dist_decomp_vortex")+"\""
    dfn = os.path.join(dpath,"dist_decomp_vortex.hdf5")
    dfp = "\""+dfn+"\""

    if not os.path.isfile(dfn):
    #Create data using solver
        print("Executing Standard Unsteady Vortex")
        estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" unsteady_standard_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + decomp_file + pts +time_str
        os.system(estr)

    vfn = os.path.join(dpath,'vortex0.hdf5')
    if not os.path.isfile(vfn):
        print("Making AnalyticalVortex")
        cvics = vortex.vics()
        cvics.STC(gamma)
        vortex.create_steady_data(cvics,npx,npy,times=times, filepath = dpath,filename = "/vortex",fdb=True)

    pfn = os.path.join(dpath,"clawres.hdf5")



    shdf5_file = h5py.File(sfn, 'r')
    swept = shdf5_file['data'][:,:,:,:]
    dhdf5_file = h5py.File(dfn, 'r')
    decomp = dhdf5_file['data'][:,:,:,:]
    vhdf5_file = h5py.File(vfn, 'r')
    avortex = vhdf5_file['data'][:,:,:,:]
    pack_file = h5py.File(pfn,'r')
    cpack = pack_file['data'][:,:,:,:]

    sfs = "%0.2e"
    # err1 = [np.amax(abs(swept[i]-decomp[i])) for i in range(len(swept))]
    err2 = [np.amax(abs(swept[i]-avortex[i])) for i in range(len(swept))]
    err3 = [np.amax(abs(decomp[i]-avortex[i])) for i in range(len(decomp))]
    err4 = [np.amax(abs(cpack[i]-avortex[i])) for i in range(len(decomp))]
    # tarr = cpack[0]-avortex[1]
    # print(err3)
    print(max(err2))
    print(max(err3))
    print(max(err4))
    xpts = np.linspace(-X/2,X/2,npx,dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    xptsn = np.linspace(-0.5,0.5,npx,dtype=np.float64)
    yptsn = np.linspace(-0.5,0.5,npy,dtype=np.float64)
    xgridn,ygridn = np.meshgrid(xptsn,yptsn,sparse=False,indexing='ij')

    pidx = -1
    fig,axes = plt.subplots(2,2)
    axes1,axes2 = axes
    ax1,ax2 = axes1
    ax3,ax4 = axes2

    ax1.contourf(xgrid,ygrid,avortex[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax2.contourf(xgridn,ygridn,cpack[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax3.contourf(xgrid,ygrid,decomp[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax4.contourf(xgrid,ygrid,swept[pidx,0,:,:],levels=20,cmap=cm.inferno)
    plt.show()

    vhdf5_file.close()
    dhdf5_file.close()
    # pack_file.close()
    if remove_file:
        os.system("rm "+sfp)
        os.system("rm "+dfp)
        os.system("rm "+vfn)




if __name__ == "__main__":
    # test_comparison_shock()
    # test_comparison_vortex(remove_file=False, generate_fig=True)
    # targs = (0,0.5,0.005,336,33.6,6,0.14,16)
    # test_swept_block(args=targs,hoststr=" --hostfile=nrg-nodes ")
    # test_standard_block(args=targs,hoststr=" --hostfile=nrg-nodes ")
    test_validate_vortex(remove_file=False)
    # test_standard_block()
    targs = (0,0.35,0.01,24,2.4,1,0,12)
    # test_swept_shock(args = targs,assertion=False,xy=False)
    # test_standard_shock(args=targs,assertion=False,xy=False)
    # targs = (0,0.5,0.005,64,6.4,1,1,32)
    # test_swept_shock(args=targs, assertion=False)
    # test_standard_shock(args=targs, assertion=False)
    pass

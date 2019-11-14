import os, sys, h5py
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
import matplotlib.animation as animation
fp = os.path.abspath(__file__)
path = os.path.dirname(fp)
sys.path.insert(0, path[:-5])
import distributed.sweep.ccore.source as source
import distributed.sweep.ccore.printer as printer
import analytical.vortex as vortex
epath = os.path.join(path[:-5],'equations')
import warnings
warnings.simplefilter('ignore')

def test_distributed_decomp_vortex(args=(1,0.01,48,0.5,10,12,1),remove_file=True,ststr=' --stationary \'true\' ',nodestr=""):
    decomp_file = "\""+os.path.join(path,"data/dist_decomp_vortex")+"\""
    sfn=os.path.join(path,"data/dist_decomp_vortex.hdf5")
    sfp = "\""+sfn+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "
    if not os.path.isfile(sfn):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+nodestr+" python "+os.path.join(path[:-5],"pst.py")+" standard_vortex "+ststr
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + decomp_file + pts +time_str
        os.system(estr)

    if 'true' in ststr:
        cvics = vortex.vics()
        cvics.Shu(1.4)
        flux_vortex = vortex.steady_vortex(cvics,npx,npy,M_o=0)[0]
        hdf5_file = h5py.File(sfn, 'r')
        data = hdf5_file['data'][:,:,:,:]
        erlist = np.asarray([np.amax(abs(dset-flux_vortex)) for dset in data])
        errorf = h5py.File(os.path.join(os.path.join(path,"data"),"decomp_err.hdf5"),'w')
        errorf.create_dataset("error",erlist.shape,data=erlist)
        errorf.close()
        hdf5_file.close()

    if remove_file:
        os.system("rm "+sfp)

def test_decomp_vortex(args=(0.5,0.01,48,0.5,10,12,4),remove_file=True):
    decomp_file = "\""+os.path.join(path,"data/decomp_vortex")+"\""
    sfp = "\""+os.path.join(path,"data/decomp_vortex.hdf5")+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "
    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_vortex --distributed \'false\' "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + decomp_file + pts +time_str
        os.system(estr)
    if remove_file:
        os.system("rm "+sfp)

def test_distributed_decomp_hde(args=(2, 48, 0.5, 10, 0.24, 5, 12, 1),remove_file=True):
    decomp_file = "\""+os.path.join(path,"data/dist_decomp_hde")+"\""
    sfp = "\""+os.path.join(path,"data/dist_decomp_hde.hdf5")+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 50*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_hde "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)
    if remove_file:
        os.system("rm "+sfp)

def test_decomp_hde(args=(1, 48, 0.5, 10, 0.24, 5, 12, 4),remove_file=True):
    decomp_file = "\""+os.path.join(path,"data/decomp_hde")+"\""
    sfp = "\""+os.path.join(path,"data/decomp_hde.hdf5")+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 50*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_hde --distributed \'false\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)
    if remove_file:
        os.system("rm "+sfp)

def test_distributed_swept_vortex(args=(1,0.01,48,0.5,40,12,1),remove_file=True,ststr=' --stationary \'true\' ',nodestr=""):
    swept_file = "\""+os.path.join(path,"data/dist_swept_vortex")+"\""
    sfn = os.path.join(path,"data/dist_swept_vortex.hdf5")
    sfp = "\""+sfn+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "

    if not os.path.isfile(sfn):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+nodestr+" python "+os.path.join(path[:-5],"pst.py")+" swept_vortex "+ststr
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + swept_file + pts +time_str
        os.system(estr)

    if 'true' in ststr:
        cvics = vortex.vics()
        cvics.Shu(1.4)
        flux_vortex = vortex.steady_vortex(cvics,npx,npy,M_o=0)[0]
        hdf5_file = h5py.File(sfn, 'r')
        data = hdf5_file['data'][:,:,:,:]
        erlist = np.asarray([np.amax(abs(dset-flux_vortex)) for dset in data])
        errorf = h5py.File(os.path.join(os.path.join(path,"data"),"swept_err.hdf5"),'w')
        errorf.create_dataset("error",erlist.shape,data=erlist)
        errorf.close()
        hdf5_file.close()

    if remove_file:
        os.system("rm "+sfp)

def test_swept_vortex(args=(0.5,0.01,48,0.5,10,12,4),remove_file=True):
    swept_file = "\""+os.path.join(path,"data/swept_vortex")+"\""
    sfp = "\""+os.path.join(path,"data/swept_vortex.hdf5")+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "
    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_vortex --distributed \'false\' "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + swept_file + pts +time_str
        os.system(estr)
    if remove_file:
        os.system("rm "+sfp)

def test_distributed_swept_hde(args=(1, 48, 0.5, 10, 0.24, 5, 12, 1),remove_file=True):
    swept_file = "\""+os.path.join(path,"data/dist_swept_hde")+"\""
    sfp = "\""+os.path.join(path,"data/dist_swept_hde.hdf5")+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 50*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_hde "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)
    if remove_file:
        os.system("rm "+sfp)

def test_swept_hde(args=(1, 48, 0.5, 10, 0.24, 5, 12, 4),remove_file=True):
    swept_file = "\""+os.path.join(path,"data/swept_hde")+"\""
    sfp = "\""+os.path.join(path,"data/swept_hde.hdf5")+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 50*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)
    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_hde --distributed \'false\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)
    if remove_file:
        os.system("rm "+sfp)


def test_comparison_hde(args=(1, 48, 0.5, 10, 0.24, 5, 12, 1),remove_file=True,generate_fig=False):
    """Use this function to compare the values obtain during a run of both solvers"""
    sfn = os.path.join(path,"data/dist_swept_hde")
    dfn = os.path.join(path,"data/dist_ndecomp_hde")
    swept_file = "\""+sfn+"\""
    decomp_file = "\""+dfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    sfp = "\""+dfn+".hdf5"+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 500*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    #Create data using solver
    estr = "mpiexec -n "+str(4)+" python "+os.path.join(path[:-5],"pst.py")+" standard_hde --distributed \'false\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)

    #Create data using solver
    estr = "mpiexec -n "+str(1)+" python "+os.path.join(path[:-5],"pst.py")+" swept_hde "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
    os.system(estr)
    #Opening the data files

    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    decomp_hdf5 = h5py.File(dfn+".hdf5", 'r')
    tsdata = swept_hdf5['data'][:,0,:,:]
    tddata = decomp_hdf5['data'][:,0,:,:]
    absa = abs(tsdata-tddata)
    max_error = np.amax(absa)
    idx = np.where(absa==max_error)
    for j,i in enumerate(absa):
        print(sum(i)/(npx*npy))
    assert np.allclose(tsdata[:-10],tddata[:-10])

    #Closing files
    swept_hdf5.close()
    decomp_hdf5.close()
    #Generating figure
    if generate_fig:
        comp_gif(dfn+".hdf5",sfn+".hdf5",filename="./hde_comp.gif")

    #Removing files
    if remove_file:
        os.system("rm "+ssfp)
        os.system("rm "+sfp)


def test_comparison_vortex(args=(5,0.01,120,0.9,10,12,1),remove_file=True,generate_fig=False):
    """Use this function to compare the values obtain during a run of both solvers"""
    sfn = os.path.join(path,"data/dist_swept_vtx")
    dfn = os.path.join(path,"data/dist_decomp_vtx")
    swept_file = "\""+sfn+"\""
    decomp_file = "\""+dfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    sfp = "\""+dfn+".hdf5"+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_vortex --distributed \'true\' --stationary \'true\'  "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + decomp_file + pts +time_str + "--gamma "+str(1.4)
    os.system(estr)

    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_vortex --distributed \'true\' --stationary \'true\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(1.4)
    os.system(estr)
    #Opening the data files

    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    decomp_hdf5 = h5py.File(dfn+".hdf5", 'r')
    tsdata = swept_hdf5['data'][:,0,:,:]
    tddata = decomp_hdf5['data'][:,0,:,:]
    max_error = np.amax(abs(tsdata-tddata))
    # assert np.allclose(tsdata,tddata)

    #Closing files
    swept_hdf5.close()
    decomp_hdf5.close()
    #Generating figure
    if generate_fig:
        comp_gif(dfn+".hdf5",sfn+".hdf5",X,Y,filename="./vtx_comp.gif")
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)
        os.system("rm "+sfp)

def test_comparison_shock(args=(2,0.01,48,0,10,12,1),remove_file=True,generate_fig=False):
    """Use this function to compare the values obtain during a run of both solvers"""
    sfn = os.path.join(path,"data/dist_swept_shock")
    dfn = os.path.join(path,"data/dist_decomp_shock")
    swept_file = "\""+sfn+"\""
    decomp_file = "\""+dfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    sfp = "\""+dfn+".hdf5"+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    #Create data using solver
    estr = "mpiexec -n "+str(4)+" python "+os.path.join(path[:-5],"pst.py")+" standard_shock --distributed \'false\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" --orient 0 "
    estr += "--hdf5 " + decomp_file + pts +time_str + "--gamma "+str(1.4)
    os.system(estr)

    #Create data using solver
    estr = "mpiexec -n "+str(4)+" python "+os.path.join(path[:-5],"pst.py")+" swept_shock --distributed \'false\' "
    estr += "-b "+str(blks)+" -a "+str(aff)+" --orient 0 "
    estr += "--hdf5 " + swept_file + pts +time_str + "--gamma "+str(1.4)
    os.system(estr)
    #Opening the data files

    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    decomp_hdf5 = h5py.File(dfn+".hdf5", 'r')
    tsdata = swept_hdf5['data'][:,0,:,:]
    tddata = decomp_hdf5['data'][:,0,:,:]
    max_error = np.amax(abs(tsdata-tddata))
    # assert np.allclose(tsdata,tddata)

    #Closing files
    swept_hdf5.close()
    decomp_hdf5.close()
    #Generating figure
    if generate_fig:
        comp_gif(dfn+".hdf5",sfn+".hdf5",filename="./shock_comp.gif")
    #Removing files
    if remove_file:
        os.system("rm "+ssfp)
        os.system("rm "+sfp)


def test_comparison_eqt(args=(1, 48, 0.5, 10, 0.24, 5, 12, 1),remove_file=True,generate_fig=False):
    """Use this function to compare the values obtain during a run of both solvers"""
    sfn = os.path.join(path,"data/dist_swept_simple")
    dfn = os.path.join(path,"data/dist_decomp_simple")
    swept_file = "\""+sfn+"\""
    decomp_file = "\""+dfn+"\""
    ssfp = "\""+sfn+".hdf5"+"\""
    sfp = "\""+dfn+".hdf5"+"\""
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 200*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_simple "
    estr += "-b "+str(blks)+" -a "+str(aff)+" --gamma 1.4 "
    estr += "--hdf5 " + decomp_file + pts +time_str
    os.system(estr)

    #Create data using solver
    estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_simple "
    estr += "-b "+str(blks)+" -a "+str(aff)+" --gamma 1.4 "
    estr += "--hdf5 " + swept_file + pts +time_str
    os.system(estr)

    #Opening the data files

    swept_hdf5 = h5py.File(sfn+".hdf5", 'r')
    decomp_hdf5 = h5py.File(dfn+".hdf5", 'r')
    tsdata = swept_hdf5['data'][:,0,:,:][:-1]
    tddata = decomp_hdf5['data'][:,0,:,:]
    assert np.allclose(tsdata,tddata)

    #Closing files
    swept_hdf5.close()
    decomp_hdf5.close()
    #Generating figure
    if generate_fig:
        comp_gif(dfn+".hdf5",sfn+".hdf5",filename="./simple_comp.gif")

    #Removing files
    if remove_file:
        os.system("rm "+ssfp)
        os.system("rm "+sfp)

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
    leftBC = (1,0,0,2.5)
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
    P[:] = [1,1,0.1,0.1,0.1]
    #Get source module
    source_mod_2D = source.build_cpu_source(os.path.join(epath,'euler.py'))
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = source.build_cpu_source(os.path.join(epath,'euler1D.py'))
    iidx = (0,slice(0,4,1),2,2)

    #Testing Flux
    dfdx,dn = source_mod_2D.dfdxy(xtest,iidx)
    dn,dfdy = source_mod_2D.dfdxy(ytest,iidx)
    dfdx = np.delete(dfdx,2)
    dfdy = np.delete(dfdy,1)
    df = source_mod_1D.fv5p(Qx,P)[2]
    assert np.allclose(df,dfdx)
    assert np.allclose(df,dfdy)

#Plottign functions
def myContour(args):
    """This funciton generates contours"""
    i,fig,ax1,ax2,ax3,xgrid,ygrid,ddata,sdata=args
    ax1.contourf(xgrid,ygrid,sdata[i,:,:],levels=20,cmap=cm.inferno)
    ax2.contourf(xgrid,ygrid,ddata[i,:,:],levels=20,cmap=cm.inferno)
    ax3.contourf(xgrid,ygrid,abs(sdata[i,:,:]-ddata[i,:,:]),levels=20,cmap=cm.inferno)

def set_lims(fig,axes,X,Y,bounds):
    """Use this function to set axis limits"""
    for i,ax in enumerate(axes):
        ax.set_ylim(-X/2, X/2)
        ax.set_xlim(-Y/2, Y/2)
        bx,by = bounds[i]
        fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(bx,by,10))

def comp_gif(decomp_file,swept_file,X,Y,filename="./comp.gif"):
    #Opening the data files
    decomp_hdf5 = h5py.File(decomp_file, 'r')
    swept_hdf5 = h5py.File(swept_file, 'r')
    tsdata = swept_hdf5['data'][:,0,:,:]
    tddata = decomp_hdf5['data'][:,0,:,:]
    adj = 10
    sdata = np.zeros((int(len(tsdata)/adj),tsdata.shape[1],tsdata.shape[2]))
    ddata = np.zeros((int(len(tddata)/adj),tddata.shape[1],tddata.shape[2]))
    for i,si in enumerate(range(0,len(tsdata)-adj,adj)):
        sdata[i,:,:] = tsdata[si,:,:]
        ddata[i,:,:] = tddata[si,:,:]
    # Meshgrid
    xpts = np.linspace(-X/2,X/2,len(sdata[0]),dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,len(ddata[0]),dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
    gs = gsc.GridSpec(2, 2)
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_title("Swept")
    ax2 = plt.subplot(gs[0,1])
    ax2.set_title("Decomp")
    ax3 = plt.subplot(gs[1,:])
    ax3.set_title('Error')
    set_lims(fig,(ax1,ax2,ax3),X,Y,((np.amin(sdata),np.amax(sdata)),(np.amin(sdata),np.amax(sdata)),(0,np.amax(abs(ddata-sdata)))))

    animate = sweep_lambda((myContour,fig,ax1,ax2,ax3,xgrid,ygrid,ddata,sdata))
    frames = len(ddata)
    anim = animation.FuncAnimation(fig,animate,frames=frames,repeat=False)
    anim.save(filename,writer="imagemagick")
    #Closing files
    swept_hdf5.close()

class sweep_lambda(object):
    """This class is a function wrapper to create kind of a pickl-able lambda function."""
    def __init__(self,args):
        self.args = args

    def __call__(self,block):
        sweep_fcn = self.args[0]
        return sweep_fcn((block,)+self.args[1:])


if __name__ == "__main__":
    # test_comparison_eqt(generate_fig=False)
    # test_comparison_shock(generate_fig=True)
    targs = (0.5,0.005,600,0.8,40,12,1)
    # test_distributed_swept_vortex(args=targs,remove_file=False,nodestr=" --hostfile=nrg-nodes ")
    test_distributed_decomp_vortex(args=targs,remove_file=False)
    # test_comparison_vortex(remove_file=False,generate_fig=True)
    # comp_gif("./pysweep/tests/data/dist_decomp_vtx.hdf5","./pysweep/tests/data/dist_swept_vtx.hdf5",1,1)
    # test_comparison_hde()
    # test_flux()

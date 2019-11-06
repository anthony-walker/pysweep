import os, sys
fp = os.path.abspath(__file__)
path = os.path.dirname(fp)

def test_distributed_decomp_vortex(args=(0.5,0.01,48,0.5,10,12,1),remove_file=True):
    decomp_file = "\""+os.path.join(path,"data/dist_decomp_vortex")+"\""
    sfp = "\""+os.path.join(path,"data/dist_decomp_vortex.hdf5")+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "
    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" standard_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + decomp_file + pts +time_str
        os.system(estr)
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

def test_distributed_decomp_hde(args=(1, 48, 0.5, 10, 0.24, 5, 12, 1),remove_file=True):
    decomp_file = "\""+os.path.join(path,"data/dist_decomp_hde")+"\""
    sfp = "\""+os.path.join(path,"data/dist_decomp_hde.hdf5")+"\""
    os.system("rm "+sfp)
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
    os.system("rm "+sfp)
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

def test_distributed_swept_vortex(args=(0.5,0.01,48,0.5,10,12,1),remove_file=True):
    swept_file = "\""+os.path.join(path,"data/dist_swept_vortex")+"\""
    sfp = "\""+os.path.join(path,"data/dist_swept_vortex.hdf5")+"\""
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "
    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python "+os.path.join(path[:-5],"pst.py")+" swept_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + swept_file + pts +time_str
        os.system(estr)
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
    os.system("rm "+sfp)
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
    os.system("rm "+sfp)
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


# def test_comparison():
#     """Use this function to compare the values obtain during a run of both solvers"""
#     savepath = "./comp_plot"
#     swept_file = "\"./tests/data/sweptc_hde\""
#     sfp = "./tests/data/swept_hde.hdf5"
#     afp = "./tests/data/analyt_hde0.hdf5"
#     decomp_file = "\"./tests/data/decompc_hde\""
#     os.system("rm "+sfp)
#     tf = 1
#     dt = 0.01
#     npx=npy= 40
#     aff = 0.5
#     X=10
#     Y=10
#     np = 16
#     bk = 10
#     Th = 373
#     Tl = 298
#     alp = 1
#     time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
#     pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)
#
#     if not os.path.isfile(sfp):
#         #Create data using solver
#         estr = "mpiexec -n "+str(np)+" python ./src/pst.py standard_hde "
#         estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
#         estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
#         os.system(estr)
#
#     if not os.path.isfile(sfp):
#         #Create data using solver
#         estr = "mpiexec -n "+str(np)+" python ./src/pst.py swept_hde "
#         estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
#         estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
#         os.system(estr)
#
#     #Opening the data files
#     swept_hdf5 = h5py.File(sfp, 'r')
#     data = swept_hdf5['data'][:,0,:,:]
#     time = np.arange(0,tf,dt)[:len(data)]
#
#     # Meshgrid
#     xpts = np.linspace(-X/2,X/2,npx,dtype=np.float64)
#     ypts = np.linspace(-Y/2,Y/2,npy,dtype=np.float64)
#     xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
#
#     fig, ax =plt.subplots()
#     ax.set_ylim(-Y/2, Y/2)
#     ax.set_xlim(-X/2, X/2)
#     ax.set_title("Density")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#
#     fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(300,375,10))
#     animate = lambda i: ax.contourf(xgrid,ygrid,data[i,:,:],levels=20,cmap=cm.inferno)
#     if isinstance(time,Iterable):
#         frames = len(tuple(time))
#         anim = animation.FuncAnimation(fig,animate,frames=frames,repeat=False)
#         anim.save(savepath+".gif",writer="imagemagick")
#     else:
#         animate(0)
#         fig.savefig(savepath+".png")
#         plt.show()
#     #Closing files
#     swept_hdf5.close()

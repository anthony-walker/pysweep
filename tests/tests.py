def test_decomp_vortex(args=(2,0.01,40,0,10,10,4)):
    decomp_file = "\"./pysweep/tests/data/decomp_vortex\""
    sfp = "./pysweep/tests/data/decomp_vortex.hdf5"
    os.system("rm "+sfp)
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)

    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python ./pysweep/pst.py standard_vortex "
        estr += "-b "+str(blks)+" -o 2 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/euler.h\" -c \"./pysweep/equations/euler.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str
        os.system(estr)

def test_decomp_hde(args=(8, 120, 0.75, 10, 0.24, 5, 12, 10)):
    savepath = "./decomp_hde_plot"
    decomp_file = "\"./pysweep/tests/data/decomp_hde\""
    sfp = "./pysweep/tests/data/decomp_hde.hdf5"
    afp = "./pysweep/tests/data/analyt_hde0.hdf5"
    analyt_file = "\"./pysweep/tests/data/analyt_hde\""
    os.system("rm "+sfp)
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    tf = 100*dt
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python ./pysweep/pst.py standard_hde "
        estr += "-b "+str(blks)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/hde.h\" -c \"./pysweep/equations/hde.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
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

def test_sweep_hde(args=(8,40,0.75,10,0.24,5,10,4)):
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
    tf = 100*dt
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "ccde mpiexec -n "+str(nps)+" python ./pysweep/pst.py swept_hde "
        estr += "-b "+str(blks)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/hde.h\" -c \"./pysweep/equations/hde.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
        os.system(estr)


def test_comparison():
    """Use this function to compare the values obtain during a run of both solvers"""
    savepath = "./comp_plot"
    swept_file = "\"./tests/data/sweptc_hde\""
    sfp = "./tests/data/swept_hde.hdf5"
    afp = "./tests/data/analyt_hde0.hdf5"
    decomp_file = "\"./tests/data/decompc_hde\""
    os.system("rm "+sfp)
    tf = 1
    dt = 0.01
    npx=npy= 40
    aff = 0.5
    X=10
    Y=10
    np = 16
    bk = 10
    Th = 373
    Tl = 298
    alp = 1
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(np)+" python ./src/pst.py standard_hde "
        estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
        os.system(estr)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(np)+" python ./src/pst.py swept_hde "
        estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
        os.system(estr)

    #Opening the data files
    swept_hdf5 = h5py.File(sfp, 'r')
    data = swept_hdf5['data'][:,0,:,:]
    time = np.arange(0,tf,dt)[:len(data)]

    # Meshgrid
    xpts = np.linspace(-X/2,X/2,npx,dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    fig, ax =plt.subplots()
    ax.set_ylim(-Y/2, Y/2)
    ax.set_xlim(-X/2, X/2)
    ax.set_title("Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(300,375,10))
    animate = lambda i: ax.contourf(xgrid,ygrid,data[i,:,:],levels=20,cmap=cm.inferno)
    if isinstance(time,Iterable):
        frames = len(tuple(time))
        anim = animation.FuncAnimation(fig,animate,frames=frames,repeat=False)
        anim.save(savepath+".gif",writer="imagemagick")
    else:
        animate(0)
        fig.savefig(savepath+".png")
        plt.show()

    #Closing files
    swept_hdf5.close()

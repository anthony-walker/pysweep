def initialize():
    #Starting timer
    start = time.time()
    #Setting global variables
    sgs.init_globals()
    #Local Constants

def sweptInput():
    #------------------INPUT DATA SETUP-------------------------$
    arr0,gargs,swargs,filename,exid,dType = decomp.read_input_file(comm)
    TSO,OPS,BS,AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    t0,tf,dt = gargs[:3]
    assert BS%(2*OPS)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
    BS = (BS,BS,1)

def standardInput():
    #------------------INPUT DATA SETUP-------------------------$
    arr0,gargs,swargs,filename,exid,dType = decomp.read_input_file(comm)
    TSO,OPS,BS,AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    sgs.TSO,sgs.OPS,sgs.BS,sgs.AF = [int(x) for x in swargs[:-1]]+[swargs[-1]]
    t0,tf,dt = gargs[:3]
    assert BS%(2*OPS)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
    BS = (BS,BS,1)

def sweptTimeSetup():
    #---------------------SWEPT VARIABLE SETUP----------------------$
    #Splits for shared array
    SPLITX = int(BS[0]/TWO)   #Split computation shift - add OPS
    SPLITY = int(BS[1]/TWO)   #Split computation shift
    MPSS = int(BS[0]/(2*OPS)-1)
    MOSS = 2*MPSS
    time_steps = int((tf-t0)/dt)  #Number of time steps
    MGST = int(TSO*(time_steps-MPSS)/(MPSS)+1)  #Global swept step  #THIS ASSUMES THAT time_steps > MOSS
    time_steps = int(MPSS*(MGST+1)/TSO+1) #Number of time steps - Add 1 for initial conditions

def standardTimeSetup():
        time_steps = int((tf-t0)/dt)  #Number of time steps



def sweptSharedArray():
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (MOSS+TSO+1,arr0.shape[0],int(sum(node_info[2:])*BS[0]),arr0.shape[2])
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,numpy.zeros(shared_shape).nbytes)
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in shared_shape]
    blocks = (bsls[0],bsls[1],blocks,bsls[3])
    GRB = True if gpu_rank is not None else False
    #Filling shared array
    if NMB:
        gsc = (slice(0,arr0.shape[1],1),slice(int(node_info[0]*BS[0]),int(node_info[1]*BS[0]),1),slice(0,arr0.shape[2],1))
        sarr[TSO-1,:,:,:] =  arr0[gsc]
        sarr[0,:,:,:] =  arr0[gsc]
    else:
        gsc = None

def standardSharedArray():
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (TSO+1,arr0.shape[0],int(sum(node_info[2:])*BS[0]+2*OPS),arr0.shape[2]+2*OPS)
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,numpy.zeros(shared_shape).nbytes)
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in shared_shape]
    blocks = (bsls[0],bsls[1],blocks,slice(bsls[3].start+OPS,bsls[3].stop-OPS,1))
    GRB = True if gpu_rank is not None else False
    #Filling shared array
    if NMB:
        gsc = (slice(0,arr0.shape[0],1),slice(int(node_info[0]*BS[0]),int(node_info[1]*BS[0]),1),slice(0,arr0.shape[2],1))
        i1,i2,i3 = gsc
        #TSO STEP and BOUNDARIES
        sarr[TSO-1,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[TSO-1,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[TSO-1,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
        #INITIAL STEP AND BOUNDARIES
        sarr[0,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[0,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[0,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
    else:
        gsc = None

def sweptCore():
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        SM,garr,Up,Down,Oct,Xb,Yb  = dcore.gpu_core(blocks,BS,OPS,gargs,GRB,MPSS,MOSS,TSO)
        mpiPool,total_cpu_block = None,None
    else:
        SM,garr = None,None
        blocks,total_cpu_block,Up,Down,Oct,Xb,Yb = dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,gargs,MPSS,TSO)
        poolSize = min(len(blocks[0]), os.cpu_count()-node_comm.Get_size()+1)
        mpiPool = mp.Pool(poolSize)
    comm.Barrier() #Ensure all processes are prepared to solve

def standardCore():
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        DecompObj,garr = dcore.gpu_core(blocks,BS,OPS,gargs,GRB,TSO)
        mpi_pool= None
    else:
        garr = None
        DecompObj,blocks= dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,GRB,TSO,gargs)
        pool_size = min(len(blocks), os.cpu_count()-node_comm.Get_size()+1)
        mpi_pool = mp.Pool(pool_size)
    functions.send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,OPS,garr,DecompObj)
    comm.Barrier() #Ensure all processes are prepared to solve

def standardSolve():
    # -------------------------------Standard Decomposition---------------------------------------------#
    node_comm.Barrier()
    cwt = 1
    for i in range(TSO*time_steps):
        functions.Decomposition(GRB,OPS,sarr,garr,blocks,mpi_pool,DecompObj)
        node_comm.Barrier()
        #Write data and copy down a step
        if (i+1)%TSO==0 and NMB:
            hdf5_data[cwt,i1,i2,i3] = sarr[TSO,:,OPS:-OPS,OPS:-OPS]
            sarr = numpy.roll(sarr,TSO,axis=0) #Copy down
            cwt+=1
        elif NMB:
            sarr = numpy.roll(sarr,TSO,axis=0) #Copy down
        node_comm.Barrier()
        #Communicate
        functions.send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,OPS,garr,DecompObj)

def sweptSolve():
    # -------------------------------SWEPT RULE---------------------------------------------#
    # -------------------------------FIRST PRISM AND COMMUNICATION-------------------------------------------#
    functions.FirstPrism(SM,GRB,Up,Yb,mpiPool,blocks,sarr,garr,total_cpu_block)
    node_comm.Barrier()
    functions.first_forward(NMB,GRB,node_comm,cluster_comm,comranks,sarr,SPLITX,total_cpu_block)
    #Loop variables
    cwt = 1 #Current write time
    gts = 0 #Initialization of global time step
    del Up #Deleting Up object after FirstPrism
    #-------------------------------SWEPT LOOP--------------------------------------------#
    step = cycle([functions.send_backward,functions.send_forward])
    for i in range(MGST):
        functions.UpPrism(GRB,Xb,Yb,Oct,mpiPool,blocks,sarr,garr,total_cpu_block)
        node_comm.Barrier()
        cwt = next(step)(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,SPLITX,gts,TSO,MPSS,total_cpu_block)
        gts+=MPSS
    #Do LastPrism Here then Write all of the remaining data
    Down.gts = Oct.gts
    functions.LastPrism(GRB,Xb,Down,mpiPool,blocks,sarr,garr,total_cpu_block)
    node_comm.Barrier()
    next(step)(cwt,sarr,hdf5_data,gsc,NMB,GRB,node_comm,cluster_comm,comranks,SPLITX,gts,TSO,MPSS,total_cpu_block)

def cleanUp():
    # Clean Up - Pop Cuda Contexts and Close Pool
    if GRB:
        cuda_context.pop()
    comm.Barrier()
    stop = time.time()
    # print(stop-stop1)
    hdf_time[0] = stop-start
    hdf5_file.close()
    #Removing input file.
    if rank==cluster_master:
        os.system("rm "+"input_file.hdf5") #remove input file
        gargs+=swargs[:4]
        if os.path.isfile('log.hdf5'):
            log_file = h5py.File('log.hdf5','a')
            shape = log_file['time'].shape[0]
            log_file['time'].resize((log_file['time'].shape[0]+1),axis=0)
            log_file['time'][shape]=stop-start
            log_file['type'].resize((log_file['type'].shape[0]+1),axis=0)
            log_file['type'][shape]=ord('s')
            log_file['args'].resize((log_file['args'].shape[0]+1),axis=0)
            log_file['args'][shape]=gargs
            log_file.close()
        else:
            log_file = h5py.File('log.hdf5','w')
            log_file.create_dataset('time',(1,),data=(stop-start),chunks=True,maxshape=(None,))
            log_file.create_dataset('type',(1,),data=(ord('s')),chunks=True,maxshape=(None,))
            log_file.create_dataset('args',(1,)+numpy.shape(gargs),data=gargs,chunks=True,maxshape=(None,)+numpy.shape(gargs))
            log_file.close()

import sys, h5py, time
from collections import Iterable

def manageLogFile(solver):
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

def createOutputFile(solver):
    """Use this function to create output file which will act as the input file as well."""
    #Create input file
    solver.hdf5 = h5py.File(solver.output, 'w', driver='mpio', comm=solver.comm)
    solver.hdf5.create_dataset("blocksize",(1,),data=(solver.blocksize[0],))
    solver.hdf5.create_dataset("share",(1,),data=(solver.share,))
    solver.hdf5.create_dataset("exid",(len(solver.exid),),data=solver.exid)
    solver.hdf5.create_dataset("globals",(len(solver.globals),),data=solver.globals)
    solver.clocktime = solver.hdf5.create_dataset("clocktime",(1,),data=0.0)
    solver.data = solver.hdf5.create_dataset("data",(solver.time_steps,)+solver.arrayShape,dtype=solver.dtype)
    if solver.clusterMasterBool:
        solver.data[0,:,:,:] = solver.initialConditions[:,:,:]
    solver.comm.Barrier()

def verbosePrint(solver,outString):
    """Use this function to print only when verbose option is specified."""
    if solver.verbose and solver.rank == solver.clusterMaster:
        print(outString)

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

def swept_write(cwt,sarr,hdf_data,gsc,gts,TSO,MPSS):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    i1,i2,i3=gsc #Unpack global tuple
    for si,i in enumerate(range(gts+1,gts+1+MPSS,1),start=TSO):
        if i%TSO==0:
            hdf_data[cwt,i1,i2,i3] = sarr[si,:,:,:]
            cwt+=1
    nte = sarr.shape[0]-MPSS
    sarr[:nte,:,:,:] = sarr[MPSS:,:,:,:]
    sarr[nte:,:,:,:] = 0
    return cwt

def read_input_file(comm,filename = 'input_file.hdf5'):
    """This function is to read the input file"""
    file = h5py.File(filename,'r',driver='mpio',comm=comm)
    gargs = tuple(file['gargs'])
    swargs = tuple(file['swargs'])
    exid = tuple(file['exid'])
    filename = ''
    for item in file['filename']:
        filename += chr(item)
    dType = ''
    for item in file['dType']:
        dType += chr(item)
    dType = numpy.dtype(dType)
    arrf = file['arr0']
    arr0 = numpy.zeros(numpy.shape(arrf), dtype=dType)
    arr0[:,:,:] = arrf[:,:,:]
    file.close()
    return arr0,gargs,swargs,filename,exid,dType

def build_gpu_source(kpath,cuda_file = "dsweep.h"):
    """Use this function to build the given and swept source module together.
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    fpath = op.join(op.join(kpath[:-len('dcore')],'ccore'),cuda_file)
    source_code = source_code_read(op.join(kpath,'gpu.h'))
    split_source_code = source_code_read(fpath).split("//!!(@#\n")
    source_code = split_source_code[0]+"\n"+source_code+"\n"+split_source_code[1]
    source_mod = SourceModule(source_code)
    return source_mod

def source_code_read(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source

def swept_constant_copy(source_mod,const_dict):
    """Use this function to copy constant args to cuda memory.
        source_mod - the source module obtained from pycuda and source code
        const_dict - dictionary of constants where the key is the global
        add_const - additional dictionary with constants. Note, they should have
                    the correct type.
    """
    #Functions to cast data to appropriate gpu types
    int_cast = lambda x:numpy.int32(x)
    float_cast = lambda x:numpy.float32(x)
    casters = {type(0.1):float_cast,type(1):int_cast}
    #Generating constants
    for key in const_dict:
        c_ptr,_ = source_mod.get_global(key)
        cst = const_dict[key]
        cuda.memcpy_htod(c_ptr,casters[type(cst)](cst))

def getDeviceAttrs(devNum=0,print_device = False):
    """Use this function to get device attributes and print them"""
    device = cuda.Device(devNum)
    dev_name = device.name()
    dev_pci_bus_id = device.pci_bus_id()
    dev_attrs = device.get_attributes()
    dev_attrs["DEVICE_NAME"]=dev_name
    if print_device:
        for x in dev_attrs:
            print(x,": ",dev_attrs[x])
    return dev_attrs

class pysweep_printer(object):
    """This function will store the master rank and print given values."""

    def __init__(self, rank,master_rank):
        self.rank = rank
        self.master = master_rank

    def __call__(self, args,p_iter=False,p_ranks=False,end="\n"):

        if (self.rank == self.master or p_ranks) and p_iter:
            if isinstance(args,Iterable):
                for item in args:
                    sys.stdout.write("[ ")
                    for si in item:
                        sys.stdout.write("%.0f"%si+", ")
                    sys.stdout.write("]\n")
            else:
                args = args,
                for item in args:
                    print(item,end=end)
        elif self.rank == self.master or p_ranks:
            print(args,end=end)

def pm(arr,i,iv=0,ps="%d"):
    for item in arr[i,iv,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write(ps%si+", ")
        sys.stdout.write("]\n")

import sys, os, h5py, time, yaml, numpy, importlib.util
from datetime import datetime
from collections import Iterable
from pycuda.compiler import SourceModule
try:
    import pycuda.driver as cuda
except Exception as e:
    print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")

def updateLogFile(solver,clocktime):
    """Use this function to update log.yaml"""
    if os.path.isfile('log.yaml'):
        with open('log.yaml','r') as f:
            previous = yaml.load(f,Loader=yaml.FullLoader)
        new = generateYamlEntry(solver,clocktime)
        previous.update(new)
        with open('log.yaml','w') as f:
            yaml.dump(previous,f)
    else:
        with open('log.yaml','w') as f:
            entry = generateYamlEntry(solver,clocktime)
            yaml.dump(entry,f)

def generateYamlEntry(obj,clocktime):
    """Use this function to generate a yaml entry."""

    rundata = {"runtime":clocktime,"swept":obj.simulation,"blocksize":obj.blocksize[0],"filename":obj.output,"verbose":obj.verbose,"share":obj.share,"globals":obj.globals,"intermediate_steps":obj.intermediate,"operating_points":obj.operating,"cpu":obj.cpuStr,"gpu":obj.gpuStr,"dtype":obj.dtypeStr,"exid":obj.exid}

    return {datetime.now().strftime("%m/%d/%Y-%H:%M:%S"):rundata}

def yamlManager(solver):
    """Use this function to manage and assign all specified yaml variables."""
    def yamlGet(key,default):
        """Use this function to get arguments from yaml file."""
        return solver.yamlFile[key] if key in solver.yamlFile else default
    document = open(solver.yamlFileName,'r')
    solver.yamlFile = yaml.load(document,Loader=yaml.FullLoader)
    #Setting simulation type - true for swept
    solver.simulation = yamlGet('swept',True)
    #Setting verbose flag
    solver.verbose = yamlGet('verbose',False)
    #setting intermediate steps
    solver.intermediate = yamlGet('intermediate_steps',1)
    #setting operating points
    solver.operating = yamlGet('operating_points',1)
    #setting blocksize
    solver.blocksize = yamlGet('blocksize',32)
    assert solver.blocksize%(2*solver.operating)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
    solver.blocksize = (solver.blocksize,solver.blocksize,1)
    #setting share
    solver.share = yamlGet('share',0.9)
    #setting exid
    solver.exid = yamlGet('exid',list())
    #setting globals
    solver.globals = yamlGet('globals',list())
    #setting cpu source
    solver.cpuStr = solver.cpu = yamlGet('cpu',"")
    #setting cpu source
    solver.gpuStr = solver.gpu = yamlGet('gpu',"")
    try:
        solver.gpu = os.path.abspath(solver.gpu)
        solver.cpu = os.path.abspath(solver.cpu)
    except Exception as e:
        pass
    #Setting output file
    solver.output = yamlGet('filename','output.hdf5')
    #Settings datatype
    solver.dtypeStr=yamlGet('dtype','float64')
    solver.dtype = numpy.dtype(solver.dtypeStr)
    #Time tuple
    spec = importlib.util.spec_from_file_location("module.step", solver.cpu)
    solver.cpu  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver.cpu)


def createOutputFile(solver):
    """Use this function to create output file which will act as the input file as well."""
    #Create input file
    solver.hdf5 = h5py.File(solver.output, 'w', driver='mpio', comm=solver.comm)
    solver.hdf5.create_dataset("blocksize",(1,),data=(solver.blocksize[0],))
    solver.hdf5.create_dataset("share",(1,),data=(solver.share,))
    solver.hdf5.create_dataset("exid",(len(solver.exid),),data=solver.exid)
    solver.hdf5.create_dataset("globals",(len(solver.globals),),data=solver.globals)
    solver.clocktime = solver.hdf5.create_dataset("clocktime",(1,),data=0.0)
    solver.data = solver.hdf5.create_dataset("data",(solver.timeSteps,)+solver.arrayShape[1:],dtype=solver.dtype)
    if solver.clusterMasterBool:
        solver.data[0,:,:,:] = solver.initialConditions[:,:,:]
    solver.comm.Barrier()

def verbosePrint(solver,outString):
    """Use this function to print only when verbose option is specified."""
    if solver.verbose and solver.clusterMasterBool:
        print(outString)

def getSolverPrint(solver):
    returnString = "\nPysweep simulation properties:\n"
    returnString += "\tsimulation type: {}\n".format("swept" if solver.simulation else "standard")

    returnString+="\tgpu source: {}\n".format(solver.source['gpu'])
    returnString+="\tcpu source: {}\n".format(solver.source['cpu'])
    returnString+="\toutput file: {}\n".format(solver.output)
    returnString+="\tverbose: {}\n".format(solver.verbose)
    returnString+="\tArray shape: {}\n".format(solver.arrayShape)
    returnString+="\tdtype: {}\n".format(solver.dtype)
    returnString+="\tshare: {}\n".format(solver.share)
    returnString+="\tblocksize: {}\n".format(solver.blocksize[0])
    returnString+="\tstencil size: {}\n".format(int(solver.operating*2+1))
    returnString+="\tintermediate time steps: {}\n".format(solver.intermediate)
    returnString+="\ttime data (t0,tf,dt): ({},{},{})\n".format(*solver.globals[:3])
    if solver.globals[5:]:
        returnString+="\tglobals: ["
        for glob in solver.globals:
            returnString+=str(glob)+","
        returnString=returnString[:-1]
        returnString+="]\n"
    if solver.exid:
        returnString+="\texcluded gpus: ["
        for exid in solver.exid:
            returnString+=str(exid)+","
        returnString=returnString[:-1]
        returnString+="]\n"
    return returnString


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

def buildGPUSource(sourcefile):
    """Use this function to build the given and swept source module together.
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    sourcefile = "\""+sourcefile+"\""
    options = ["-DPYSWEEP_GPU_SOURCE={}".format(sourcefile),]
    kernel = os.path.join(os.path.abspath(os.path.dirname(__file__)),"pysweep.cu")
    sourceCode = readSourceCode(kernel)
    return SourceModule(sourceCode,options=options)


def readSourceCode(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source

def copySweptConstants(source_mod,const_dict):
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

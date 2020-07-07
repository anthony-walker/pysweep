import sys, os, h5py, time, yaml, numpy
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

    rundata = {"runtime":clocktime,"swept":obj.simulation,"blocksize":int(obj.blocksize[0]),"filename":obj.output,"verbose":obj.verbose,"share":float(obj.share),"globals":obj.globals,"intermediate_steps":obj.intermediate,"operating_points":obj.operating,"cpu":obj.cpuStr,"gpu":obj.gpuStr,"dtype":obj.dtypeStr,"exid":obj.exid}
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
    #Loading CPU module
    solver.loadCPUModule()


def createOutputFile(solver):
    """Use this function to create output file which will act as the input file as well."""
    #Create input file
    solver.hdf5 = h5py.File(solver.output, 'w', driver='mpio', comm=solver.comm)
    solver.hdf5.create_dataset("blocksize",(1,),data=(solver.blocksize[0],))
    solver.hdf5.create_dataset("share",(1,),data=(solver.share,))
    solver.hdf5.create_dataset("exid",(len(solver.exid),),data=solver.exid)
    solver.hdf5.create_dataset("globals",(len(solver.globals),),data=solver.globals)
    solver.clocktime = solver.hdf5.create_dataset("clocktime",(1,),data=0.0)
    solver.data = solver.hdf5.create_dataset("data",solver.arrayShape,dtype=solver.dtype)
    if solver.clusterMasterBool:
        solver.data[0,:,:,:] = solver.initialConditions[:,:,:]
    solver.comm.Barrier()

def verbosePrint(solver,outString):
    """Use this function to print only when verbose option is specified."""
    if solver.verbose and solver.clusterMasterBool:
        print(outString)

def debugPrint(solver,outString):
    """Use this function to print for debugging."""
    if solver.clusterMasterBool:
        print(outString)

def getSolverPrint(solver):
    returnString = "\nPysweep simulation properties:\n"
    returnString += "\tsimulation type: {}\n".format("swept" if solver.simulation else "standard")
    try:
        returnString+="\tnumber of processes: {}\n".format(solver.comm.Get_size())
    except Exception as e:
        pass
    returnString+="\tgpu source: {}\n".format(solver.gpuStr)
    returnString+="\tcpu source: {}\n".format(solver.cpuStr)
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

def sweptWrite(cwt,solver):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    iv,ix,iy = solver.globalBlock #Unpack global tuple
    for i in range(solver.globalTimeStep%solver.intermediate+solver.intermediate-solver.subtraction,solver.maxPyramidSize+solver.intermediate-solver.subtraction,solver.intermediate):
        solver.data[cwt,iv,ix,iy] = solver.sharedArray[i,:,:,:]
        cwt+=1
    nte = solver.sharedShape[0]-solver.maxPyramidSize
    solver.sharedArray[:nte,:,:,:] = solver.sharedArray[solver.maxPyramidSize:,:,:,:]
    # solver.sharedArray[nte:,:,:,:] = 0 #This shouldn't be necessary - debugging
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

def copyConstants(source_mod,const_dict):
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

def standardWrite(cwt,solver):
    """Use this function to write standard data out."""
    solver.nodeComm.Barrier()
    #Write data and copy down a step
    if (solver.globalTimeStep)%solver.intermediate==0 and solver.nodeMasterBool:
        #Unpacking globalBlock
        iv,ix,iy = solver.globalBlock #Unpack global tuple
        solver.data[cwt,iv,ix,iy] = solver.sharedArray[solver.intermediate,:,solver.operating:-solver.operating,solver.operating:-solver.operating]
        cwt+=1
    solver.nodeComm.Barrier()
    #Update CPU shared data
    for block in solver.blocks:
        writeblock,readblock = block
        it,iv,ix,iy = writeblock
        for i in range(solver.intermediate):
            solver.sharedArray[i,iv,ix,iy] = solver.sharedArray[i+1,iv,ix,iy]
    #Update GPU shared data
    if solver.gpuBool:
        it,iv,ix,iy = solver.gpuBlock
        for i in range(solver.intermediate):
            solver.sharedArray[i,iv,ix,iy] = solver.sharedArray[i+1,iv,ix,iy]
    solver.nodeComm.Barrier()
    return cwt

def systemOutDebug(solver,array=None):
    """Use this function to write data out with standard out for more clear debugging."""
    
    def writeOut(arr):
        for row in arr:
            sys.stdout.write("[")
            for item in row:
                if item == 0:
                    sys.stdout.write("\033[1;36m")
                else:
                    sys.stdout.write("\033[1;31m")
                sys.stdout.write("%.0f, "%item)
            sys.stdout.write("]\n")
    
    if solver.clusterMasterBool: 
            ci = 0
            while ci != -1:
                if array is None:
                    writeOut(solver.sharedArray[ci,0,:,:])
                else:
                    writeOut(array[ci,0,:,:])
                ci = int(input())
    solver.comm.Barrier()
    
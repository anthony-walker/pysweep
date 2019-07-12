#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
from pysweep_lambda import sweep_lambda
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


def constant_copy(source_mod,ps,grid_size,block_size,ops,add_const=None):
    """Use this function to copy constant args to cuda memory.
        source_mod - the source module obtained from pycuda and source code
        ps - shape of the 4D array (t,v,x,y)
        block_size -  size of blocks to run on GPU
        grid_size -  size of GPU grid
        ops - number of atomic operations
        add_const - additional dictionary with key(global):(cast_fcn, data)
        Note, the key should match the global in the source_mod and the cast_fcn
        should convert the data to the appropriate type.
        e.g.  int_cast = lambda x:np.int32(x) add_const = {SOME_CONST:(int_cast,myInt)}
    """
    #Functions to cast data to appropriate gpu types
    int_cast = lambda x:np.int32(x)
    float_cast = lambda x:np.float32(x)

    #Dictionary for constants
    const_dict = dict()
    #Generating constants
    MSS = min(block_size[2:])/(2*ops) #Max swept step
    NV = ps[1]  #Number of variables in given initial array
    SGIDS = block_size[0]*block_size[1]
    VARS =  SGIDS*grid_size[0]*grid_size[1]
    TIMES = VARS*NV
    const_dict['MSS'] = (int_cast,MSS)
    const_dict['NV'] = (int_cast,NV)
    const_dict['SGIDS'] = (int_cast,SGIDS)
    const_dict['VARS'] = (int_cast,VARS)
    const_dict['TIMES'] = (int_cast,TIMES)
    const_dict['OPS'] = (int_cast,ops)
    const_dict['SGNVS'] = (int_cast,SGIDS*NV)
    for key in const_dict:
        c_ptr,_ = source_mod.get_global(key)
        cuda.memcpy_htod(c_ptr,const_dict[key][0](const_dict[key][1]))
    return MSS,NV,SGIDS,VARS,TIMES,const_dict

def UpPyramid(arr, ops):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    ops -  the number of atomic operations
    """
    plane_shape = np.shape(arr)
    iidx = list(np.ndindex(plane_shape[2:]))
    #Bounds
    lb = 0
    ub = [plane_shape[2],plane_shape[3]]
    #Going through all swept steps
    # pts = [iidx]    #This is strictly for debugging
    while ub[0] > lb and ub[1] > lb:
        lb += ops
        ub = [x-ops for x in ub]
        iidx = [x for x in iidx if x[0]>=lb and x[1]>=lb and x[0]<ub[0] and x[1]<ub[1]]
        if iidx:
            step(arr,iidx,0)
            # pts.append(iidx) #This is strictly for debuggings
    # return pts #This is strictly for

def Octahedron(arr,  ops):
    """This is the step(s) in between UpPyramid and DownPyramid."""
    pass

def DownPyramid(arr, ops):
    """This is the ending inverted pyramid."""
    pass

#--------------------------------NOT USED CURRENTLY-----------------------------
def edges(arr,ops,shape_adj=-1):
    """Use this function to generate boolean arrays for edge handling."""
    mask = np.zeros(arr.shape[:shape_adj], dtype=bool)
    mask[(arr.ndim+shape_adj)*(slice(ops, -ops),)] = True
    return mask

def dummy_fcn(arr):
    """This is a testing function for arch_speed_comp."""
    iidx = np.ndindex(np.shape(arr))
    for i,idx in enumerate(iidx):
        arr[idx] *= i
        # print(i,value)
    return arr

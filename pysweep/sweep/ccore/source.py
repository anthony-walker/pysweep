#Programmer: Anthony Walker
#This file contains all the necessary resources to build the source modules
#Other imports
import numpy as np
#Cuda imports
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
#System imports
import os.path as op

def build_gpu_source(kpath):
    """Use this function to build the given and swept source module together.
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    cuda_file = "dsweep.h"
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
    int_cast = lambda x:np.int32(x)
    float_cast = lambda x:np.float32(x)
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

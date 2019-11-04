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
import inspect
import importlib.util as iu

def build_cpu_source(cpu_source):
    """Use this function to build source module from cpu code."""
    module_name = cpu_source.split("/")[-1]
    module_name = module_name.split(".")[0]
    spec = iu.spec_from_file_location(module_name, cpu_source)
    source_mod = iu.module_from_spec(spec)
    spec.loader.exec_module(source_mod)
    return source_mod

def build_gpu_source(kernel_source,name):
    """Use this function to build the given and swept source module together.
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    if 'dc' in name:
        cuda_file = "dsweep.h"
    else:
        cuda_file = "nsweep.h"
    file = inspect.getfile(build_cpu_source)
    fname = file.split("/")[-1]
    fpath = op.abspath(inspect.getabsfile(build_cpu_source))[:-len(fname)]+cuda_file
    source_code = source_code_read(kernel_source)
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

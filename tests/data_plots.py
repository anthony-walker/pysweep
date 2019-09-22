#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import h5py




def create_combined_hdf5(nfn="/home/walkanth/pysweep/results/sdcomp.hdf5",sfn="swept-vortex",dfn="decomp-vortex",direc="/home/walkanth/pysweep/results"):
    """Use this function to create a combined hdf set"""
    files = os.listdir(direc)
    print(files)
    hf = h5py.File(files[0],'r')
    # time = hf['time']
    keys = list(hf.keys())
    keys = ['AF', 'BS', 'array_size', 'data', 'time']
    if 'data' in keys:
        keys.remove('data')

    new_file = h5py.File(nfn,'w')
    dataset = new_file.create_dataset('data',(len(files),len(keys)+1,))
    for k,file in enumerate(files):
        cf = hf = h5py.File(file,'r')
        dataset[k,0] = True if 'swept' in file else False
        for i,key in enumerate(keys,start=1):
            dataset[k,i] = cf[key][0]
    hf.close()
    new_file.close()
    # print(time)
create_combined_hdf5()

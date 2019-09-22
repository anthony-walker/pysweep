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
from master import controller

def create_test_files():
    """Use this function to create test files."""
    self = controller()
    hdfend = ".hdf5"
    names = ['./results/swept_vortex','./results/decomp_vortex']
    for n in names:
        for i, affinity in enumerate(self.affs):
            for j, block in enumerate(self.blocks):
                for k, size in enumerate(self.sizes):
                    print(i,j,k)
                    hf = h5py.File(n+str(i)+str(j)+str(k)+hdfend,'w')
                    aff = hf.create_dataset('AF',(1,))
                    aff[0] = affinity
                    bs = hf.create_dataset('BS',(1,))
                    bs[0] = block
                    arr_size = hf.create_dataset('array_size',(4,))
                    arr_size[:] = (500,4,size,size)
                    tme = hf.create_dataset('time',(1,))
                    if 'decomp' in n:
                        tme[0] = 10000/size/affinity/block*1.5
                    else:
                        tme[0] = 10000/size/affinity/block
                    hf.close()


def create_combined_hdf5(nfn="sdcomp.hdf5",sfn="swept_vortex",dfn="decomp_vortex",direc="/home/walkanth/pysweep/results/"):
    """Use this function to create a combined hdf set"""
    files = os.listdir(direc)
    hf = h5py.File(direc+files[0],'r')
    keys = list(hf.keys())
    if 'data' in keys:
        keys.remove('data')
    new_file = h5py.File(nfn,'w')
    dataset = new_file.create_dataset('data',(len(files),len(keys)+1,))
    for k,file in enumerate(files):
        cf = h5py.File(direc+file,'r')
        dataset[k,0] = True if 'swept' in file else False
        for i,key in enumerate(keys,start=1):
            if key == 'array_size':
                dataset[k,i] = cf[key][2]
            else:
                dataset[k,i] = cf[key][0]
    hf.close()
    new_file.close()

def create_swept_data(fn="sdcomp.hdf5"):
    hf = h5py.File(fn,'r')
    keys = list(hf.keys())
    data = hf['data']
    sdata = np.zeros((int(len(data[:,:])/2),4))
    ddata = np.zeros((int(len(data[:,:])/2),4))
    st = 0
    dt = 0
    for i,row in enumerate(data[:,:]):
        if row[0]:
            sdata[st,:] = data[i,1:]
            st+=1
        else:
            ddata[dt,:] = data[i,1:]
            dt+=1
    return sdata,ddata

def create_swept_plot(cases=(0.5,0.6,0.7,0.8),cv2=12):
    """Use this function to create figures"""
    sdata,ddata = create_swept_data()
    sdata[:,0] = np.round(sdata[:,0],1)
    ddata[:,0] = np.round(ddata[:,0],1)
    #Keep affinity constant:

    index = 1
    sdata = sdata[sdata[:,index].argsort()]
    ddata = ddata[ddata[:,index].argsort()]

    # sset = sorted(list(set(sdata[:,index])))
    # sdict = dict()
    # ddict = dict()
    #
    # for key in sset:
    #     sdict[key]= (list(),list(),list(),list())
    #     ddict[key]= (list(),list(),list(),list())
    #
    # for row in sdata:
    #     sdict[row[index]][0].append(row[0])
    #     sdict[row[index]][1].append(row[1])
    #     sdict[row[index]][2].append(row[2])
    #     sdict[row[index]][3].append(row[3])
    #
    # for row in ddata:
    #     ddict[row[index]][0].append(row[0])
    #     ddict[row[index]][1].append(row[1])
    #     ddict[row[index]][2].append(row[2])
    #     ddict[row[index]][3].append(row[3])
    #
    # fig = plt.figure()
    # sax = fig.add_subplot(1,2,1)
    # dax = fig.add_subplot(1,2,2)
    # colors = ['blue','red','green','yellow','black','magenta','cyan']
    # for i,key in enumerate(sset):
    #     print(sdict[key][3])
    #     print(sdict[key][1])
    #     sax.scatter(sdict[key][2],sdict[key][3],color=colors[i])
    #     dax.scatter(ddict[key][2],ddict[key][3],color=colors[i])
    # plt.show()
# create_test_files()
# create_combined_hdf5()
create_swept_plot()
#

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
    # print(self.blocks,self.sizes)
    hdfend = ".hdf5"
    names = ['./results/swept_vortex','./results/decomp_vortex']
    for n in names:
        for i, affinity in enumerate(self.affs):
            for j, block in enumerate(self.blocks):
                for k, size in enumerate(self.sizes):
                    # print(i,j,k)
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

def create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(12,),idx1=1):
    """Use this function to create figures"""
    sdata,ddata = create_swept_data()
    sdata[:,0] = np.round(sdata[:,0],1)
    ddata[:,0] = np.round(ddata[:,0],1)
    idxs = [0,1,2,3]
    idxs.remove(idx0)
    idxs.remove(idx1)
    idxx,idxy=idxs
    fig = plt.figure()
    sax = fig.add_subplot(1,2,1)
    dax = fig.add_subplot(1,2,2)
    plt.subplots_adjust(wspace=0.5,bottom=0.25)
    colors = ['blue','red','green','yellow','black','magenta','cyan']
    labels = ['Affinity','Block Size','Array Size','Time']
    xlab = labels[idxx]
    ylab = labels[idxy]
    sax.set_title('Swept')
    sax.set_ylabel(ylab)
    sax.set_xlabel(xlab)
    dax.set_title('Standard')
    dax.set_ylabel(ylab)
    dax.set_xlabel(xlab)
    clab = labels[idx0]
    clab2 = labels[idx1]
    leglist = list()
    fig.suptitle(clab+" Performance Comparison with Constant "+clab2+" ("+str(cv2[0])+")")
    maxy = 0
    #Keep affinity constant:
    ct = 0
    for c1 in cv1:
        leglist.append(str(c1))
        for c2 in cv2:
            xl = list()
            yl = list()
            for row in sdata:
                if row[idx0] == c1 and row[idx1]==c2:
                    xl.append(row[idxx])
                    yl.append(row[idxy])
                    if row[idxy] > maxy:
                        maxy = row[idxy]
            Z = [(x,y) for x,y in sorted(zip(xl,yl))]
            if Z:
                xl = list()
                yl = list()
                for z in Z:
                    xl.append(z[0])
                    yl.append(z[1])
                sax.plot(xl,yl,color=colors[ct],marker='o')
                ct+=1
    sax.legend(leglist,ncol=len(leglist),bbox_to_anchor=(1.25, -0.3),loc="lower center")
    #Keep affinity constant:
    ct = 0
    for c1 in cv1:
        for c2 in cv2:
            xl = list()
            yl = list()
            for row in ddata:
                if row[idx0] == c1 and row[idx1]==c2:
                    xl.append(row[idxx])
                    yl.append(row[idxy])
                    if row[idxy] > maxy:
                        maxy = row[idxy]
            Z = [(x,y) for x,y in sorted(zip(xl,yl))]
            if Z:
                xl = list()
                yl = list()
                for z in Z:
                    xl.append(z[0])
                    yl.append(z[1])
                dax.plot(xl,yl,color=colors[ct],marker='o')
                ct+=1
    dax.set_ylim(0,np.ceil(maxy))
    sax.set_ylim(0,np.ceil(maxy))
    clb2str = str(cv2[0])
    clb2str = clb2str.replace('.','')
    fig.savefig('./figures/'+clab[:5]+"_"+xlab[:5]+"_"+clab2[:5]+clb2str)
    return fig

def create_case_plots():
    """This function creates the plots of interest for the paper"""
    #Constant block size plots
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(8,),idx1=1)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(12,),idx1=1)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(16,),idx1=1)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(24,),idx1=1)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(32,),idx1=1)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(8,),idx1=1)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(12,),idx1=1)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(16,),idx1=1)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(24,),idx1=1)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(32,),idx1=1)
    #Constant Affinity Plots
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(0.5,),idx1=0)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(0.6,),idx1=0)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(0.7,),idx1=0)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(0.8,),idx1=0)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(0.9,),idx1=0)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(1.0,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(0.5,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(0.6,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(0.7,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(0.8,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(0.9,),idx1=0)
    create_swept_plot(cv1=(96,192,288,284,480),idx0=2,cv2=(1.0,),idx1=0)

    #Constant Array Size Plot
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(96,),idx1=2)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(192,),idx1=2)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(288,),idx1=2)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(384,),idx1=2)
    create_swept_plot(cv1=(0.5,0.6,0.7,0.8,0.9,1),idx0=0,cv2=(480,),idx1=2)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(96,),idx1=2)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(192,),idx1=2)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(288,),idx1=2)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(384,),idx1=2)
    create_swept_plot(cv1=(8,12,16,24,32),idx0=1,cv2=(480,),idx1=2)


if __name__ == "__main__":
    create_case_plots()

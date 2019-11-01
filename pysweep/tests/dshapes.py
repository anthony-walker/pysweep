#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os
sys.path.insert(0, './pysweep')
import numpy as np
from itertools import cycle
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from sweep.dcore import block

colors = ['red','dodgerblue','orange','green']
symbols = cycle(['o','o','o','o'])
elev = 55
bsx = 12
ops = 1
nd = 4
npx = nd*bsx

def UpPyramids():
    fig = plt.figure()
    fig.suptitle("UpPyramid Phase",fontweight="bold")
    ax = fig.add_subplot(1,1,1,projection='3d',elev=elev)
    ax.set_title("Calculation Step",y=1.08)
    upsets = block.create_dist_up_sets((bsx,bsx,1),ops)
    ts = len(upsets)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    ax.set_zlim3d([0,10])
    ax.get_zaxis().set_ticks([0,10])
    ax.get_xaxis().set_ticks([0,int(bsx*nd/2),int(bsx*nd)])
    ax.get_yaxis().set_ticks([0,int(bsx*nd/2),int(bsx*nd)])

    # print(upsets)
    ct = 0
    for bdx in range(int(npx/bsx)):
        cs = next(symbols)
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker=cs,alpha=1,edgecolor='black',facecolor=colors[ct])

        ct+=1
    f = lambda x,y,z: mplot3d.proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    ax.legend(['Rank 0','Rank 1','Rank 2','Rank 3'],ncol=4,loc="lower center", bbox_to_anchor=f(25,0,-21),
          bbox_transform=ax.transData)
    plt.show()

if __name__ == "__main__":
    UpPyramids()

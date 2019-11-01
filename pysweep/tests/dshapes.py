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

colors = ['dodgerblue','orange','dodgerblue','orange','dodgerblue','orange']
symbols = cycle(['o','o','o','o'])
elev = 45
bsx = 12
ops = 2
nd = 4
npx = nd*bsx
MPSS = int(bsx/(2*ops)-1)
slb = 0
sub = npx

def make_block(ax,start,length,width,height,sc="blue"):
    """Use this function to make a block surface"""
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+length])
    zz = np.ones(np.shape(xx))*start[2]
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors='black')
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+length])
    zz = np.ones(np.shape(xx))*(start[2]+height)
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors='black')
    xx, yy = np.meshgrid([start[0],start[0]+height], [start[1],start[1]+length])
    zz = np.ones(np.shape(xx))*(start[0])
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors='black')
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors='black')
    xx, yy = np.meshgrid([start[0],start[0]+height], [start[1],start[1]+length])
    zz = np.ones(np.shape(xx))*(start[2]+length)
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors='black')
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors='black')


def UpPyramids():
    fig = plt.figure()
    # fig.suptitle("UpPyramid",fontweight="bold")
    ax = fig.add_subplot(1,1,1,projection='3d',elev=elev)
    ax.set_title("UpPyramid",fontweight='bold',y=1.1)
    upsets = block.create_dist_up_sets((bsx,bsx,1),ops)
    yset,xset = block.create_dist_bridge_sets((bsx,bsx,1),ops,MPSS)
    ts = len(upsets)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    zlim = 20
    bsh = int(bsx/2)
    ax.set_zlim3d([0,zlim])
    ax.set_ylim3d([slb-bsh,sub+bsh])
    ax.set_xlim3d([slb-bsh,sub+bsh])
    ax.get_zaxis().set_ticks([0,10])
    ax.get_xaxis().set_ticks([0,int(bsx*nd/2),int(bsx*nd)])
    ax.get_yaxis().set_ticks([0,int(bsx*nd/2),int(bsx*nd)])
    make_block(ax,(0,0,0),bsx,bsx,1)
    # x1,y1 = zip(*upsets[0])
    # z0 = np.zeros(len(x1))
    # z1 = np.ones(len(x1))
    # z = np.concatenate((z0,z1))
    # x = np.concatenate((x1,x1))
    # y = np.concatenate((y1,y1))
    # ct = 0
    # ax.plot_trisurf(x,y,z,alpha=1,color=colors[ct])

    # ms = 6
    # ct = 0
    # lines = []
    # for bdx in range(int(npx/bsx)):
    #     cs = next(symbols)
    #     for bdy in range(0,int(npx/bsx),1):
    #         x1 = []
    #         y1 = []
    #         z1 = []
    #         for k,uset in enumerate(upsets,start=1):
    #             for i,j in uset:
    #                 tx = i+bdx*bsx
    #                 ty = j+bdy*bsx
    #                 if tx>=slb and tx<=sub and ty>=slb and ty<=sub:
    #                     x1.append(tx)
    #                     y1.append(ty)
    #                     z1.append(k)
    #         line = ax.plot_trisurf(x1,y1,z1,alpha=1,color=colors[ct])
    #     lines.append(line)
    #     ct+=1
    # l1 = ax.plot_surface(xx1,yy1,zz1,color='red',alpha=.5)
    # l2 = ax.plot_surface(xx2,yy2,zz2,color='green',alpha=.5)
    # f = lambda x,y,z: mplot3d.proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    # leg1 = plt.legend(lines[:2],[' GPU   ',' CPU   '],markerscale=2,ncol=2,loc="lower center", bbox_to_anchor=f(25.5,0,-19),
    #       bbox_transform=ax.transData)
    # ax.add_artist(leg1)
    # fl1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o',markersize=ms,markeredgecolor='black')
    # fl2 = mpl.lines.Line2D([0],[0], linestyle="none", c='green', marker = 'o',markersize=ms,markeredgecolor='black')
    # leg2 = ax.legend([fl1,fl2], ['node 1','node 2'],ncol=4,markerscale=1.1, loc="lower center", bbox_to_anchor=f(25,0,-25), bbox_transform=ax.transData)
    # ax.gca().add_artist(leg1)
    plt.savefig('UpPyramid.png')
    ax.set_title("Y-Bridge",fontweight='bold',y=1.1)

    # ct = 0
    # lines = []
    # for bdx in range(0,int(npx/bsx),1):
    #     cs = next(symbols)
    #     for bdy in range(-1,int(npx/bsx)+1,1):
    #         x1 = []
    #         y1 = []
    #         z1 = []
    #         for k,uset in enumerate(yset,start=1):
    #             for i,j in uset:
    #                 tx = i+bdx*bsx
    #                 ty = j+bdy*bsx+bsx/2
    #                 if tx>=slb and tx<=sub and ty>=slb and ty<=sub:
    #                     x1.append(tx)
    #                     y1.append(ty)
    #                     z1.append(k)
    #         line = ax.scatter(x1,y1,z1,marker=cs,alpha=1,edgecolor='black',s=12,facecolor=colors[ct])
    #     lines.append(line)
    #     ct+=1
    # plt.savefig('YBridge.png')
    # ax.set_title("X-Bridge",fontweight='bold',y=1.1)
    # ct = 0
    # lines = []
    # for bdx in range(-1,int(npx/bsx)+1,1):
    #     cs = next(symbols)
    #     for bdy in range(-1,int(npx/bsx)+1,1):
    #         x1 = []
    #         y1 = []
    #         z1 = []
    #         for k,uset in enumerate(xset,start=1):
    #             for i,j in uset:
    #                 tx = i+bdx*bsx+bsx/2
    #                 ty = j+bdy*bsx
    #                 if tx>=slb and tx<=sub and ty>=slb and ty<=sub:
    #                     x1.append(tx)
    #                     y1.append(ty)
    #                     z1.append(k)
    #         line = ax.scatter(x1,y1,z1,marker=cs,alpha=1,edgecolor='black',s=12,facecolor=colors[ct])
    #     lines.append(line)
    #     ct+=1
    # plt.savefig('YBridge.png')
    # ax.set_title("X-Bridge",fontweight='bold',y=1.1)
    # # plt.savefig('XBridge.png')

    plt.show()


def filter_set(set,idx):
    new_set = []
    for T in set:
        if T[idx]>=slb and T[idx]<=sub:
            new_set.append(T)
    return new_set


if __name__ == "__main__":
    UpPyramids()

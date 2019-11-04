#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from sweep.ncore import block

def OCT_Fig():
    ops = 1
    bsx = 12
    npx = int(2*bsx+2*ops)
    ts = 10
    shape = (ts,npx+bsx/2,npx+bsx/2)
    bs = (bsx,bsx,1)

    upsets = block.create_up_sets(bs,ops)
    downsets = block.create_down_sets(bs,ops)
    octsets = downsets+upsets[1:]
    bridge_sets, bridge_slices = block.create_bridge_sets(bs,ops,len(upsets))
    time_set = np.arange(0,len(upsets))
    colors = ['blue','red','green','orange']
    # --------------------------- UpPyramid  Phase----------------------------
    #Create figure
    elev = 55
    fig = plt.figure()
    fig.suptitle("Octahedron Phase",fontweight="bold")
    plt.subplots_adjust(hspace=0.5)
    ax,ax1 = OctPlot(fig,shape,npx,bsx,upsets,colors,ops,elev,bridge_sets)
    OctPhase(ax,shape,npx,bsx,octsets,colors)
    CMP2(ax1,shape,npx,bsx,octsets,colors,ops)
    fig.savefig("Octaheron.png")
    # plt.show()

def CMP2(ax,shape,npx,bsx,upsets,colors,ops):
    """This is the first communication phase"""
    ct=0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets,start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx+bsx/2)
                    y1.append(j+bdy*bsx+bsx/2)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1

    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(upsets):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j >= bsx/2+ops and i >= bsx/2+ops:
                zc.append((i-bsx/2+ops,j-bsx/2+ops))

            if j >= bsx/2+ops:
                zy.append((i-bsx/2+ops,j-bsx/2+ops))

            if i >= bsx/2+ops:
                zx.append((i-bsx/2+ops,j-bsx/2+ops))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)


    x1 = []
    y1 = []
    z1 = []
    for k,uset in enumerate(c_set,start=1):
        for i,j in uset:
            x1.append(i)
            y1.append(j)
            z1.append(k)
    ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor='orange')

    ct = 1
    for bdy in range(1,int(npx/bsx)+1):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(y_set,start=1):
            for i,j in uset:
                x1.append(i+bsx*bdy)
                y1.append(j)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=2

    ct = 2
    for bdy in range(1,int(npx/bsx)+1):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(y_set,start=1):
            for i,j in uset:
                x1.append(j)
                y1.append(i+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=1

def OctPhase(ax,shape,npx,bsx,upsets,colors):
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets,start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx+bsx/2)
                    y1.append(j+bdy*bsx+bsx/2)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1
    f = lambda x,y,z: mplot3d.proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    leg = ax.legend(['Rank 0','Rank 1','Rank 2','Rank 3'],ncol=4,loc="lower center", bbox_to_anchor=f(25,0,-21),
          bbox_transform=ax.transData)
    for i,lh in enumerate(leg.legendHandles):
        lh.set_alpha(1)
        lh.set_color(colors[i])
        lh.set_edgecolor('black')

def OctPlot(fig,shape,npx,bsx,upsets,colors,ops,elev,bridge_sets,alpha=0.25):
    """This is the first communication phase"""
    ax = fig.add_subplot(2,1,1,projection='3d',elev=elev)
    ax1 = fig.add_subplot(2,1,2,projection='3d',elev=elev)
    ax.set_title("Calculation Step",y=1.04)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    ax.get_zaxis().set_ticks([0,4,8])
    ax.get_xaxis().set_ticks([0,15,30])
    ax.get_yaxis().set_ticks([0,15,30])
    ax.set_xlim3d(0,shape[1])
    ax.set_ylim3d(0,shape[1])
    ax.set_zlim3d(0,shape[0])

    ax1.set_title("Communication Step",y=1.04)
    ax1.set_xlabel("\n\nX")
    ax1.set_ylabel("\n\nY")
    ax1.set_zlabel("t")
    ax1.get_zaxis().set_ticks([0,4,8])
    ax1.get_xaxis().set_ticks([0,15,30])
    ax1.get_yaxis().set_ticks([0,15,30])
    ax1.set_xlim3d(0,shape[1])
    ax1.set_ylim3d(0,shape[1])
    ax1.set_zlim3d(0,shape[0])

    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ct+=1
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(bridge_sets[0],start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx+bsx/2)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ct+=1
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(bridge_sets[1],start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx+bsx/2)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
            ct+=1

    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(upsets):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j <= bsx/2+ops and i < bsx/2+ops:
                zc.append((i,j))

            if j <= bsx/2+ops:
                zy.append((i,j))

            if i <= bsx/2+ops:
                zx.append((i,j))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)

    x1 = []
    y1 = []
    z1 = []
    for k,uset in enumerate(c_set):
        for i,j in uset:
            x1.append(i+npx-2*ops)
            y1.append(j+npx-2*ops)
            z1.append(k)
    ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[0],alpha=alpha)
    ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[0],alpha=alpha)

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                x1.append(i+npx-2*ops)
                y1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ct+=1

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                y1.append(i+npx-2*ops)
                x1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ct+=2

    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(bridge_sets[1]):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j <= bsx/2+ops and i < bsx/2+ops:
                zc.append((i,j))

            if j <= bsx/2+ops:
                zy.append((i,j))

            if i <= bsx/2+ops:
                zx.append((i,j))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set,start=1):
            for i,j in uset:
                x1.append(i+npx-2*ops)
                y1.append(j+bdy*bsx+bsx/2)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ct+=1

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set,start=1):
            for i,j in uset:
                y1.append(i+npx-2*ops)
                x1.append(j+bdy*bsx+bsx/2)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ax1.scatter(x1,y1,z1,marker="o",edgecolor='black',facecolor=colors[ct],alpha=alpha)
        ct+=2

    return ax,ax1




def BR_Fig():
    ops = 1
    bsx = 12
    npx = int(2*bsx+2*ops)
    ts = 10
    shape = (ts,npx+bsx/2,npx+bsx/2)
    bs = (bsx,bsx,1)

    upsets = create_up_sets(bs,ops)
    downsets = create_down_sets(bs,ops)
    octsets = downsets+upsets[1:]
    bridge_sets, bridge_slices = create_bridge_sets(bs,ops,len(upsets))
    time_set = np.arange(0,len(upsets))
    colors = ['blue','red','green','orange']
    # --------------------------- UpPyramid  Phase----------------------------
    #Create figure
    elev = 55
    fig = plt.figure()
    fig.suptitle("Bridge Phase",fontweight="bold")
    plt.subplots_adjust(hspace=0.5)
    BridgePlot(fig,shape,npx,bsx,upsets,colors,ops,elev,bridge_sets)
    fig.savefig("Bridge.png")

def BridgePlot(fig,shape,npx,bsx,upsets,colors,ops,elev,bridge_sets):
    """This is the first communication phase"""
    ax = fig.add_subplot(2,1,1,projection='3d',elev=elev)
    ax1 = fig.add_subplot(2,1,2,projection='3d',elev=elev)
    ax.set_title("Calculation Step",y=1.04)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    ax.get_zaxis().set_ticks([0,4,8])
    ax.get_xaxis().set_ticks([0,15,30])
    ax.get_yaxis().set_ticks([0,15,30])
    ax.set_xlim3d(0,shape[1])
    ax.set_ylim3d(0,shape[1])
    ax.set_zlim3d(0,shape[0])

    ax1.set_title("Communication Step",y=1.04)
    ax1.set_xlabel("\n\nX")
    ax1.set_ylabel("\n\nY")
    ax1.set_zlabel("t")
    ax1.get_zaxis().set_ticks([0,4,8])
    ax1.get_xaxis().set_ticks([0,15,30])
    ax1.get_yaxis().set_ticks([0,15,30])
    ax1.set_xlim3d(0,shape[1])
    ax1.set_ylim3d(0,shape[1])
    ax1.set_zlim3d(0,shape[0])

    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(bridge_sets[0],start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx+bsx/2)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(bridge_sets[1],start=1):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx+bsx/2)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1

    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(upsets):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j <= bsx/2+ops and i < bsx/2+ops:
                zc.append((i,j))

            if j <= bsx/2+ops:
                zy.append((i,j))

            if i <= bsx/2+ops:
                zx.append((i,j))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)

    x1 = []
    y1 = []
    z1 = []
    for k,uset in enumerate(c_set):
        for i,j in uset:
            x1.append(i+npx-2*ops)
            y1.append(j+npx-2*ops)
            z1.append(k)
    ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[0])
    ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[0])

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                x1.append(i+npx-2*ops)
                y1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=1

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                y1.append(i+npx-2*ops)
                x1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=2
    f = lambda x,y,z: mplot3d.proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    ax.legend(['Rank 0','Rank 1','Rank 2','Rank 3'],ncol=4,loc="lower center", bbox_to_anchor=f(25,0,-21),
          bbox_transform=ax.transData)
    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(bridge_sets[1]):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j <= bsx/2+ops and i < bsx/2+ops:
                zc.append((i,j))

            if j <= bsx/2+ops:
                zy.append((i,j))

            if i <= bsx/2+ops:
                zx.append((i,j))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set,start=1):
            for i,j in uset:
                x1.append(i+npx-2*ops)
                y1.append(j+bdy*bsx+bsx/2)
                z1.append(k)
        ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=1

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set,start=1):
            for i,j in uset:
                y1.append(i+npx-2*ops)
                x1.append(j+bdy*bsx+bsx/2)
                z1.append(k)
        ax1.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=2


def UP_Fig():
    ops = 1
    bsx = 12
    npx = int(2*bsx+2*ops)
    ts = 10
    shape = (ts,npx+bsx/2,npx+bsx/2)
    bs = (bsx,bsx,1)

    upsets = create_up_sets(bs,ops)
    downsets = create_down_sets(bs,ops)
    octsets = downsets+upsets[1:]
    bridge_sets, bridge_slices = create_bridge_sets(bs,ops,len(upsets))
    time_set = np.arange(0,len(upsets))
    colors = ['blue','red','green','orange']
    # --------------------------- UpPyramid  Phase----------------------------
    #Create figure
    elev = 55
    fig = plt.figure()
    fig.suptitle("UpPyramid Phase",fontweight="bold")
    plt.subplots_adjust(hspace=0.5)
    UpPyramidPhase(fig,shape,npx,bsx,upsets,colors,elev)
    CMP1(fig,shape,npx,bsx,upsets,colors,ops,elev)
    fig.savefig("UpPyramid.png")


def CMP1(fig,shape,npx,bsx,upsets,colors,ops,elev):
    """This is the first communication phase"""
    ax = fig.add_subplot(2,1,2,projection='3d',elev=elev)
    ax.set_title("Communication Step",y=1.04)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    ax.get_zaxis().set_ticks([0,4,8])
    ax.get_xaxis().set_ticks([0,15,30])
    ax.get_yaxis().set_ticks([0,15,30])
    ax.set_xlim3d(0,shape[1])
    ax.set_ylim3d(0,shape[1])
    ax.set_zlim3d(0,shape[0])
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1

    #Make edge and corner sets
    y_set = list()
    x_set = list()
    c_set = list()
    for k,uset in enumerate(upsets):
        zc = []
        zx = []
        zy = []
        for i,j in uset:
            if  j <= bsx/2+ops and i <= bsx/2+ops:
                zc.append((i,j))

            if j <= bsx/2+ops:
                zy.append((i,j))

            if i <= bsx/2+ops:
                zx.append((i,j))

        c_set.append(zc)
        y_set.append(zy)
        x_set.append(zx)

    x1 = []
    y1 = []
    z1 = []
    for k,uset in enumerate(c_set):
        for i,j in uset:
            x1.append(i+npx-2*ops)
            y1.append(j+npx-2*ops)
            z1.append(k)
    ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[0])

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                x1.append(i+npx-2*ops)
                y1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=1

    ct = 0
    for bdy in range(int(npx/bsx)):
        x1 = []
        y1 = []
        z1 = []
        for k,uset in enumerate(x_set):
            for i,j in uset:
                y1.append(i+npx-2*ops)
                x1.append(j+bdy*bsx)
                z1.append(k)
        ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
        ct+=2

def UpPyramidPhase(fig,shape,npx,bsx,upsets,colors,elev):
    ax = fig.add_subplot(2,1,1,projection='3d',elev=elev)
    ax.set_title("Calculation Step",y=1.08)
    ax.set_xlabel("\n\nX")
    ax.set_ylabel("\n\nY")
    ax.set_zlabel("t")
    ax.get_zaxis().set_ticks([0,4,8])
    ax.get_xaxis().set_ticks([0,15,30])
    ax.get_yaxis().set_ticks([0,15,30])
    ax.set_xlim3d(0,shape[1])
    ax.set_ylim3d(0,shape[1])
    ax.set_zlim3d(0,shape[0])
    ct = 0
    for bdx in range(int(npx/bsx)):
        for bdy in range(int(npx/bsx)):
            x1 = []
            y1 = []
            z1 = []
            for k,uset in enumerate(upsets):
                for i,j in uset:
                    x1.append(i+bdx*bsx)
                    y1.append(j+bdy*bsx)
                    z1.append(k)
            ax.scatter(x1,y1,z1,marker="o",alpha=1,edgecolor='black',facecolor=colors[ct])
            ct+=1
    f = lambda x,y,z: mplot3d.proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    ax.legend(['Rank 0','Rank 1','Rank 2','Rank 3'],ncol=4,loc="lower center", bbox_to_anchor=f(25,0,-21),
          bbox_transform=ax.transData)

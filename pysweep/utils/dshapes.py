#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os
sys.path.insert(0, '/home/walkanth/swept-project/')
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
from matplotlib.patches import FancyArrowPatch


colors = ['dodgerblue','orange','dodgerblue','orange','dodgerblue','orange']
symbols = cycle(['o','o','o','o'])

elev2 =45
azim2=45

anch_box = (-15,-5,5)
alps = [0,0.1,1]
lnc = 1
bsx = 12
bsh = int(bsx/2)
ops = 1
nd = 4
ndy = 2
npx = nd*bsx
npy = ndy*bsx
MPSS = int(bsx/(2*ops)-1)
slb = 0
sub = npx
zlim = 15
ylims = [slb-bsh,npy+bsh]
xlims = [slb-bsh,sub+bsh]
zlims = [0,zlim]
mscal = 20
lwidth = 1.5
asty = "->"
node_alpha = 0.2
upsets = block.create_dist_up_sets((bsx,bsx,1),ops)
downsets = block.create_dist_down_sets((bsx,bsx,1),ops)
yset,xset = block.create_dist_bridge_sets((bsx,bsx,1),ops,MPSS)
ms = 6
scale = 1.2
palp = 0.25


def make_node_surfaces(ax,colors,spacing):
    """Use this method to make node base colors"""
    for i,c in enumerate(colors):
        xx, yy = np.meshgrid([i*spacing-bsh,(i+1)*spacing-bsh], [-bsh,npy+bsh])
        zz = np.zeros(np.shape(xx))*-1
        ax.plot_surface(xx,yy,zz,color=c,alpha=node_alpha)

def make_node_tri_surfaces(ax,colors):
    """Use this method to make node base colors"""
    spacing=10
    n1 = ax.plot_trisurf(np.linspace(xlims[0],xlims[1],spacing),np.linspace(ylims[0],ylims[1],spacing),np.zeros(spacing),color=colors[0],alpha=0.2)
    n2 = ax.plot_trisurf(np.linspace(xlims[0],xlims[1],spacing),np.linspace(ylims[0],ylims[1],spacing),np.zeros(spacing),color=colors[1],alpha=0.2)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def create_arrow_axes(ax):
    """This function replaces the normal axes with arrows."""
    aclr = "k"
    fac = 4
    a = Arrow3D([xlims[0], xlims[0]], [ylims[0], ylims[0]],[zlims[0]-2, zlims[1]+2], mutation_scale=mscal, lw=lwidth, arrowstyle=asty, color=aclr)
    b = Arrow3D([xlims[0], xlims[0]], [ylims[0], ylims[1]+3],[zlims[0], zlims[0]], mutation_scale=mscal, lw=lwidth, arrowstyle=asty, color=aclr)
    c = Arrow3D([xlims[0], xlims[1]+5], [ylims[0], ylims[0]],[zlims[0], zlims[0]], mutation_scale=mscal, lw=lwidth, arrowstyle=asty, color=aclr)

    ax.text(xlims[1],ylims[0]-3,0,'X')
    ax.text(xlims[0]-4,ylims[1],0,'Y')
    ax.text(xlims[0]-3,ylims[0]+2,zlims[1],'t')
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(c)
    ax.axis('off')

def make_block(ax,start,length,width,height,sc="blue",alp=1,edges=True):
    """Use this function to make a block surface"""
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+width])
    ec = 'black' if edges else None
    zz = np.ones(np.shape(xx))*start[2]
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[1],start[1]+width])
    zz = np.ones(np.shape(xx))*(start[2]+height)
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = np.meshgrid([start[2],start[2]+height], [start[1],start[1]+width])
    zz = np.ones(np.shape(xx))*(start[0])
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors=ec,alpha=alp)
    zz = np.ones(np.shape(xx))*(start[1])
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[2],start[2]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = np.meshgrid([start[2],start[2]+height], [start[1],start[1]+width])
    zz = np.ones(np.shape(xx))*(start[0]+length)
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors=ec,alpha=alp)
    zz = np.ones(np.shape(xx))*(start[1]+width)
    xx, yy = np.meshgrid([start[0],start[0]+length], [start[2],start[2]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors=ec,alpha=alp)

def plot_uppyramid(ax,start=0,edges=True,alp=1):
    ct = 0
    for i in range(nd):
        for j in range(ndy):
            for k in range(1,len(upsets)+1):
                make_block(ax,(k*ops+i*bsx,k*ops+j*bsx,k+start),bsx-2*ops*k,bsx-2*ops*k,1,sc=colors[ct],edges=edges,alp=alp)
        ct+=1

def plot_ybridges(ax,start=0,edges=True,alp=1):
    # Create Y-Bridges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(k*ops+i*bsx,bsx-ops-k*ops,k+start),bsx-2*ops*k,2*ops*(k)+ops,1,sc=colors[ct],edges=edges,alp=alp)
        ct+=1
    #Close Edge Y Bridges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(k*ops+i*bsx,0,k+start),bsx-2*ops*k,ops*(k),1,sc=colors[ct],edges=edges,alp=alp)
        ct+=1
    #Far Edge Y Bridges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(k*ops+i*bsx,npy-ops*k,k+start),bsx-2*ops*k,ops*(k),1,sc=colors[ct],edges=edges,alp=alp)
        ct+=1

def plot_xbridges(ax,start=0,edges=True,alp=1):
    # Create X-Bridges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(bsh+i*bsx-k*ops,bsh+k*ops,k+start),2*ops*(k)+ops,bsx-2*ops*k,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1
    #Close edges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(bsh+i*bsx-k*ops,0,k+start),2*ops*(k),bsh-ops*k,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1
    #Far Edges
    ct = 0
    for i in range(nd):
        for k in range(1,len(yset)+1):
            make_block(ax,(bsh+i*bsx-k*ops,npy-bsh+k*ops,k+start),2*ops*(k),bsh-ops*k,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1

def plot_octahedrons(ax,start=0,edges=True,alp=1):
    ct = 0
    dsl = len(downsets)
    for i in range(nd):
        for j in range(ndy):
            for k in range(1,len(upsets)+dsl+1):
                if k <= dsl:
                    make_block(ax,(bsh-k*ops+i*bsx,bsh-k*ops+j*bsx,k+start),2*ops*k,2*ops*k,1,sc=colors[ct],alp=alp,edges=edges)
                else:
                    make_block(ax,((k-dsl)*ops+i*bsx,(k-dsl)*ops+j*bsx,k+start),bsx-2*ops*(k-dsl),bsx-2*ops*(k-dsl),1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1

def plot_comm1(ax,start=0,edges=True,alp=1):
    ct = 0
    for i in range(nd):
        for k in range(1,len(upsets)+1):
            make_block(ax,(0+i*bsx,0,k+start),bsh-ops*k,npy,1,sc=colors[ct],alp=alp,edges=edges)
            make_block(ax,(bsh+i*bsx+k*ops,0,k+start),bsh-ops*k,npy,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1

def plot_comm2(ax,start=0,edges=True,alp=1):
    ct = 0
    usl = len(upsets)
    for i in range(nd):
        for k in range(1,len(upsets)+1):
            if k <= len(upsets):
                make_block(ax,(i*bsx,0,k+start),bsx,npy,1,sc=colors[ct],alp=alp,edges=edges)
        for k,nz in enumerate(range(usl,len(yset)+usl+1,1)):
                make_block(ax,(0+i*bsx,0,nz+start),bsh-ops*k,npy,1,sc=colors[ct],alp=alp,edges=edges)
                make_block(ax,(bsh+i*bsx+k*ops,0,nz+start),bsh-ops*k,npy,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1

def plot_dwp(ax,start=0,edges=True,alp=1):
    ct = 0
    usl = len(upsets)
    dsl = len(downsets)
    for i in range(nd):
        for j in range(ndy):
            for k,nz in enumerate(range(usl+1,len(yset)+usl+1,1),1):
                if k <= dsl:
                    make_block(ax,(bsh-k*ops+i*bsx,bsh-k*ops+j*bsx,nz+start),2*ops*k,2*ops*k,1,sc=colors[ct],alp=alp,edges=edges)
        ct+=1

def staxf(elev=40,azim=35):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d',elev=elev,azim=azim)
    ax.set_zlim3d((zlims[0],zlims[1]+5))
    ax.set_ylim3d(ylims)
    ax.set_xlim3d(xlims)
    ax.get_zaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.xaxis._axinfo['juggled'] = (0,0,0)
    ax.yaxis._axinfo['juggled'] = (1,1,1)
    ax.zaxis._axinfo['juggled'] = (2,2,2)
    make_node_surfaces(ax,['red','green'],(npx+bsx)/ndy)
    create_arrow_axes(ax)
    f = lambda tup: mplot3d.proj3d.proj_transform(tup[0],tup[1],tup[2], ax.get_proj())[:2]
    fl1 = mpl.lines.Line2D([0],[0], linestyle="none", c='dodgerblue', marker = 'o',markersize=ms)
    fl2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o',markersize=ms)
    fl3 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o',markersize=ms,alpha=node_alpha)
    fl4 = mpl.lines.Line2D([0],[0], linestyle="none", c='green', marker = 'o',markersize=ms,alpha=node_alpha)
    leg1 = ax.legend([fl1,fl2,fl3,fl4],[' GPU   ',' CPU    ','node 1','node 2'],markerscale=scale,ncol=lnc,loc="lower left", bbox_to_anchor=f(anch_box), #-40,0,-5
          bbox_transform=ax.transData)
    return fig, ax


def Up1():
    fig,ax = staxf()
    plot_uppyramid(ax,0)
    plt.savefig('UpPyramid1.png',bbox_inches='tight')
    plt.close(fig)

def Y1():
    fig,ax = staxf()
    plot_uppyramid(ax,edges=False,alp=palp)
    plot_ybridges(ax)
    plt.savefig('YBridge1.png',bbox_inches='tight')
    plt.close(fig)

def Comm1():
    fig,ax = staxf()
    plot_comm1(ax)
    plt.savefig('Comm1.png',bbox_inches='tight')
    plt.close(fig)

def X1():
    fig,ax = staxf()
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax)
    plt.savefig('XBridge1.png',bbox_inches='tight')
    plt.close(fig)

def Oct1():
    fig,ax = staxf(elev=elev2,azim=azim2)
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax,edges=False,alp=palp)
    plot_octahedrons(ax)
    plt.savefig('Octahedron1.png',bbox_inches='tight')
    plt.close(fig)

def Y2():
    fig,ax = staxf(elev=elev2,azim=azim2)
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax,edges=False,alp=palp)
    plot_octahedrons(ax,edges=False,alp=palp)
    plot_ybridges(ax,start=len(upsets))
    plt.savefig('YBridge2.png',bbox_inches='tight')
    plt.close(fig)

def Comm2():
    fig,ax = staxf(elev=elev2,azim=azim2)
    plot_comm2(ax)
    plt.savefig('Comm2.png',bbox_inches='tight')

def X2():
    fig,ax = staxf(elev=elev2,azim=azim2)
    plot_comm2(ax,edges=False,alp=palp)
    plot_xbridges(ax,start=len(upsets))
    plt.savefig('XBridge2.png',bbox_inches='tight')
    plt.close(fig)

def DWP1():
    fig,ax = staxf(elev=elev2,azim=azim2)
    plot_comm2(ax,edges=False,alp=palp)
    plot_xbridges(ax,start=len(upsets),edges=False,alp=palp)
    plot_dwp(ax)
    plt.savefig("DownPyramid1.png",bbox_inches='tight')



if __name__ == "__main__":
    Up1()
    Y1()
    Comm1()
    X1()
    palp = 0.1
    Oct1()
    Y2()
    Comm2()
    X2()
    DWP1()

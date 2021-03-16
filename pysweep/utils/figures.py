#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os, numpy, warnings
import pysweep.core.block as block
from itertools import cycle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch



colors = ['dodgerblue','orange','dodgerblue','orange','dodgerblue','orange']
symbols = cycle(['o','o','o','o'])

#Angle
elev2 =45
azim2=45
#Angle
gelev=40
gazim=35

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
upsets = block.createUpPyramidSets((bsx,bsx,1),ops)
downsets = block.createDownPyramidSets((bsx,bsx,1),ops)
yset,xset = block.createBridgeSets((bsx,bsx,1),ops,MPSS)
USL = len(upsets)
YSL = len(yset)
XSL = len(xset)
DSL = len(downsets)
H = USL+DSL
ms = 6
scale = 1.2
palp = 0.25
#Global
gfig = plt.figure()
gax = gfig.add_subplot(1,1,1,projection='3d',elev=gelev,azim=gazim)

def switchColorScheme():
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white","grid.color":"white","savefig.facecolor":"#333333","savefig.edgecolor":"#333333","axes.facecolor":"#333333"})

def make_node_surfaces(ax,colors,spacing):
    """Use this method to make node base colors"""
    for i,c in enumerate(colors):
        xx, yy = numpy.meshgrid([i*spacing-bsh,(i+1)*spacing-bsh], [-bsh,npy+bsh])
        zz = numpy.zeros(numpy.shape(xx))*-1
        ax.plot_surface(xx,yy,zz,color=c,alpha=node_alpha)

def make_node_tri_surfaces(ax,colors):
    """Use this method to make node base colors"""
    spacing=10
    n1 = ax.plot_trisurf(numpy.linspace(xlims[0],xlims[1],spacing),numpy.linspace(ylims[0],ylims[1],spacing),numpy.zeros(spacing),color=colors[0],alpha=0.2)
    n2 = ax.plot_trisurf(numpy.linspace(xlims[0],xlims[1],spacing),numpy.linspace(ylims[0],ylims[1],spacing),numpy.zeros(spacing),color=colors[1],alpha=0.2)


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
    xx, yy = numpy.meshgrid([start[0],start[0]+length], [start[1],start[1]+width])
    ec = 'black' if edges else None
    zz = numpy.ones(numpy.shape(xx))*start[2]
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = numpy.meshgrid([start[0],start[0]+length], [start[1],start[1]+width])
    zz = numpy.ones(numpy.shape(xx))*(start[2]+height)
    ax.plot_surface(xx,yy,zz,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = numpy.meshgrid([start[2],start[2]+height], [start[1],start[1]+width])
    zz = numpy.ones(numpy.shape(xx))*(start[0])
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors=ec,alpha=alp)
    zz = numpy.ones(numpy.shape(xx))*(start[1])
    xx, yy = numpy.meshgrid([start[0],start[0]+length], [start[2],start[2]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors=ec,alpha=alp)
    xx, yy = numpy.meshgrid([start[2],start[2]+height], [start[1],start[1]+width])
    zz = numpy.ones(numpy.shape(xx))*(start[0]+length)
    ax.plot_surface(zz,yy,xx,color=sc,edgecolors=ec,alpha=alp)
    zz = numpy.ones(numpy.shape(xx))*(start[1]+width)
    xx, yy = numpy.meshgrid([start[0],start[0]+length], [start[2],start[2]+height])
    ax.plot_surface(xx,zz,yy,color=sc,edgecolors=ec,alpha=alp)

def plot_uppyramid(ax,start=0,edges=True,alp=1,L=USL+1):
    color = 'dodgerblue'
    for k in range(1,L):
        for i in range(nd): #nd
            for j in range(ndy): #ndy
                make_block(ax,(k*ops+i*bsx,k*ops+j*bsx,k+start),bsx-2*ops*k,bsx-2*ops*k,1,sc=color,edges=edges,alp=alp)
            color = 'dodgerblue' if color == 'orange' else 'orange'
             

def plot_ybridges(ax,start=0,edges=True,alp=1,L=YSL+1):
    # Create Y-Bridges
    color = 'dodgerblue'
    for k in range(1,L):
        for i in range(nd):
            #Interior
            make_block(ax,(k*ops+i*bsx,bsx-ops-k*ops,k+start),bsx-2*ops*k,2*ops*(k)+ops,1,sc=color,edges=edges,alp=alp)
            #Close Edge Y Bridges
            make_block(ax,(k*ops+i*bsx,0,k+start),bsx-2*ops*k,ops*(k),1,sc=color,edges=edges,alp=alp)
            #Far Edge Y Bridges
            make_block(ax,(k*ops+i*bsx,npy-ops*k,k+start),bsx-2*ops*k,ops*(k),1,sc=color,edges=edges,alp=alp)
            color = 'dodgerblue' if color == 'orange' else 'orange'

def plot_xbridges(ax,start=0,edges=True,alp=1,L=XSL+1):
    # Create X-Bridges
    color = 'dodgerblue'
    for k in range(1,L):
        for i in range(nd):
            make_block(ax,(bsh+i*bsx-k*ops,bsh+k*ops,k+start),2*ops*(k)+ops,bsx-2*ops*k,1,sc=color,alp=alp,edges=edges)
            #Close Edges
            make_block(ax,(bsh+i*bsx-k*ops,0,k+start),2*ops*(k),bsh-ops*k,1,sc=color,alp=alp,edges=edges)
            #Far Edges
            make_block(ax,(bsh+i*bsx-k*ops,npy-bsh+k*ops,k+start),2*ops*(k),bsh-ops*k,1,sc=color,alp=alp,edges=edges)
            color = 'dodgerblue' if color == 'orange' else 'orange'

def plot_octahedrons(ax,start=0,edges=True,alp=1,L = USL+DSL+1):
    color = 'dodgerblue'
    dsl = len(downsets)
    for k in range(1,L,1):
        for i in range(nd):
            for j in range(ndy):
                if k <= dsl:
                    make_block(ax,(bsh-k*ops+i*bsx,bsh-k*ops+j*bsx,k+start),2*ops*k,2*ops*k,1,sc=color,alp=alp,edges=edges)
                else:
                    make_block(ax,((k-dsl)*ops+i*bsx,(k-dsl)*ops+j*bsx,k+start),bsx-2*ops*(k-dsl),bsx-2*ops*(k-dsl),1,sc=color,alp=alp,edges=edges)
            color = 'dodgerblue' if color == 'orange' else 'orange'

def plot_comm1(ax,start=0,edges=True,alp=1,L=USL+1):
    color = 'dodgerblue'
    for k in range(1,len(upsets)+1):
        for i in range(nd):
            make_block(ax,(0+i*bsx,0,k+start),bsh-ops*k,npy,1,sc=color,alp=alp,edges=edges)
            make_block(ax,(bsh+i*bsx+k*ops,0,k+start),bsh-ops*k,npy,1,sc=color,alp=alp,edges=edges)
            color = 'dodgerblue' if color == 'orange' else 'orange'

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

def plot_dwp(ax,start=0,edges=True,alp=1,L=USL+1):
    color = 'dodgerblue'
    for k,nz in enumerate(range(USL+1,L+USL,1)):
        for i in range(nd):
            for j in range(ndy):
                if k <= DSL:
                    make_block(ax,(bsh-k*ops+i*bsx,bsh-k*ops+j*bsx,nz+start),2*ops*k,2*ops*k,1,sc=color,alp=alp,edges=edges)
            color = 'dodgerblue' if color == 'orange' else 'orange'

def staxf(ax,elev=40,azim=35):
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
    return ax

def Up1(ax,name='UpPyramid1.pdf'):
    ax = staxf(ax,elev=gelev,azim=gazim)
    plot_uppyramid(ax,0)
    plt.savefig(name,bbox_inches='tight')
    return ax
    
def Y1(ax,name='YBridge1.pdf'):
    ax = staxf(ax)
    plot_uppyramid(ax,edges=False,alp=palp)
    plot_ybridges(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def Comm1(ax,name='Comm1.pdf'):
    ax = staxf(ax)
    plot_comm1(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def X1(ax,name='XBridge1.pdf'):
    ax = staxf(ax)
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def Oct1(ax,name='Octahedron1.pdf'):
    ax = staxf(ax)
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax,edges=False,alp=palp)
    plot_octahedrons(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def Y2(ax,name='YBridge2.pdf'):
    ax = staxf(ax)
    plot_comm1(ax,edges=False,alp=palp)
    plot_xbridges(ax,edges=False,alp=palp)
    plot_octahedrons(ax,edges=False,alp=palp)
    plot_ybridges(ax,start=len(upsets))
    plt.savefig(name,bbox_inches='tight')
    return ax

def Comm2(ax,name='Comm2.pdf'):
    ax = staxf(ax)
    plot_comm2(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def X2(ax,name='XBridge2.pdf'):
    ax = staxf(ax)
    plot_comm2(ax,edges=False,alp=palp)
    plot_xbridges(ax,start=len(upsets))
    plt.savefig(name,bbox_inches='tight')
    return ax

def DWP1(ax,name="DownPyramid1.pdf"):
    ax = staxf(ax)
    plot_comm2(ax,edges=False,alp=palp)
    plot_xbridges(ax,start=len(upsets),edges=False,alp=palp)
    plot_dwp(ax)
    plt.savefig(name,bbox_inches='tight')
    return ax

def createAll():
    """Use this function to generate all paper figures."""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d',elev=gelev,azim=gazim)
    ax.set_facecolor("#333333")
    Up1(ax)
    ax.clear()
    Y1(ax)
    ax.clear()
    Comm1(ax)
    X1(ax)
    ax.clear()
    Oct1(ax)
    ax.clear()
    Y2(ax)
    ax.clear()
    Comm2(ax)
    ax.clear()
    X2(ax)
    ax.clear()
    DWP1(ax)
    ax.clear()


def createSubFigurePlots():
    fig = plt.figure()
    axes = [fig.add_subplot(2,2,i,projection='3d',elev=gelev,azim=gazim) for i in range(1,5,1)]
    letters = ["a","b","c","d"]
    #First subplot
    name = "SubsPlot1.pdf"
    fcns1 = [Up1,Y1,Comm1,X1]
    for i,fcn in enumerate(fcns1):
        fcn(axes[i],name=name)
        axes[i].set_title("({})".format(letters[i]))

    #Second subplot
    name2 = "SubsPlot2.pdf"
    fcns2 = [Oct1,X2,Y2,DWP1]
    for i,fcn in enumerate(fcns2):
        axes[i].clear()
        fcn(axes[i],name=name2)    

def createPresentationGif():
    
    frames = 9*USL+2
    anim = animation.FuncAnimation(gfig,plotStep,frames)
    anim.save("SweptProcess.gif",writer="imagemagick",fps=3)
    
def plotStep(i):
    global gax
    gax.clear()
    gax.set_facecolor("#333333")
    gax = staxf(gax)
    if i<USL+1:
        plot_uppyramid(gax,0,L=i+1)
    elif i<=2*USL:
        idx = i-YSL
        plot_uppyramid(gax,0)#,alp=palp,edges=False)
        plot_ybridges(gax,L=idx+1)
    elif i==2*USL+1:
        plot_comm1(gax)
    elif i<=3*USL+1:
        plot_comm1(gax,alp=palp,edges=False)
        plot_xbridges(gax,L=i%(H))
    elif i<=5*USL+1:
        idx = i-3*USL-1
        plot_comm1(gax,alp=palp,edges=False)
        plot_xbridges(gax,alp=palp,edges=False)
        plot_octahedrons(gax,L=idx+1)
    elif i<=6*USL+1:
        idx = i-5*USL-1
        plot_comm1(gax,edges=False,alp=palp)
        plot_xbridges(gax,edges=False,alp=palp)
        plot_octahedrons(gax,edges=False,alp=palp)
        plot_ybridges(gax,start=USL,L=idx+1)
    elif i==6*USL+2:
        plot_comm2(gax)
    elif i<=7*USL+2:
        idx = i-6*USL-2
        plot_comm2(gax,alp=palp,edges=False)
        plot_xbridges(gax,start=USL,L=idx+1)
    else:
        idx = i-7*USL-2
        plot_comm2(gax,edges=False,alp=palp)
        plot_xbridges(gax,start=USL,edges=False,alp=palp)
        plot_dwp(gax,L=idx+1)

def arrowAxes(ax,xlims,ylims,zlims):
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

def numericalImpactImage():
    color = 'dodgerblue'
    fig = plt.figure()
    elev=40
    azim=35
    start=0
    xyl = [0,npx//4]
    zlmax = 8
    ax = fig.add_subplot(1,1,1,projection='3d',elev=45,azim=45)
    ax.set_zlim3d([0,zlmax])
    ax.set_ylim3d(xyl)
    ax.set_xlim3d(xyl)
    ax.get_zaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.xaxis._axinfo['juggled'] = (0,0,0)
    ax.yaxis._axinfo['juggled'] = (1,1,1)
    ax.zaxis._axinfo['juggled'] = (2,2,2)
    arrowAxes(ax,xyl,xyl,[0,zlmax-2])
    for k in range(USL+1):
        for i in range(1): #nd
            for j in range(1): #ndy
                make_block(ax,(k*ops+i*bsx,k*ops+j*bsx,k+start),bsx-2*ops*k,bsx-2*ops*k,1,sc=color,edges=True,alp=1)
            color = 'dodgerblue' if color == 'orange' else 'orange'
    plt.savefig("NumericalImpact-1.pdf")
    ax.view_init(0, 10)
    plt.savefig("NumericalImpact-2.pdf")
    ax.view_init(90, 90)
    plt.savefig("NumericalImpact-3.pdf")

if __name__ == "__main__":
    switchColorScheme()
    # createAll()
    
    # createSubFigurePlots()
    # createPresentationGif()
    numericalImpactImage()
    

    
import h5py,numpy
#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib import animation, rc

def heatContourAx(ax,data,Lx,Ly):
    shape = numpy.shape(data)
    x = numpy.linspace(0,Lx,shape[0])
    y = numpy.linspace(0,Ly,shape[1])
    X,Y = numpy.meshgrid(x,y)
    ax.contourf(X,Y,data,cmap=cm.inferno,vmin=-1,vmax=1)

def eulerContourAx(ax,data,Lx,Ly):
    shape = numpy.shape(data)
    x = numpy.linspace(-Lx,Lx,shape[0])
    y = numpy.linspace(-Ly,Ly,shape[1])
    X,Y = numpy.meshgrid(x,y)
    ax.contourf(X,Y,data,cmap=cm.inferno,vmin=0.4,vmax=1)

def createSurface(data,tid,Lx,Ly,Lz,xlab="X",ylab="Y",filename="surface.pdf",gif=False,elev=45,azim=25,gmod=1):
    """Use this as a function for create gif."""
    global fig,ax,X,Y,gifData,LZ,LX,LY
    fig =  plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(cm.ScalarMappable(cmap=cm.magma),ax=ax,boundaries=numpy.linspace(0,Lz,10))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    LX=Lx
    LY=Ly
    LZ=Lz
    #Create x and y data
    shape = numpy.shape(data)
    x = numpy.linspace(-Lx,Lx,shape[1])
    y = numpy.linspace(-Ly,Ly,shape[2])
    X,Y = numpy.meshgrid(x,y)
    if gif:
        gifDataRange = range(0,len(data),gmod)
        gifData = numpy.zeros((len(gifDataRange),)+numpy.shape(data)[1:])
        for i,j in enumerate(gifDataRange):
            gifData[i,:,:] = data[j,:,:]
        frames = len(gifData)
        anim = animation.FuncAnimation(fig,animateSurface,frames)
        anim.save(filename.split(".")[0]+".gif",writer="imagemagick")
    else:
        ax.surface(X,Y,data[tid])
        plt.savefig(filename)

def animateSurface(i): 
    ax.cla()
    ax.set_zlim(-LZ,LZ)
    ax.set_xlim(-LX,LX)
    ax.set_ylim(-LY,LY)
    ax.plot_surface(X,Y,gifData[i],cmap=cm.magma,vmin=-LZ, vmax=LZ)

def createContourf(data,tid,Lx,Ly,Lz,xlab="X",ylab="Y",filename="contour.pdf",gif=False,gmod=1,LZn=None):
    """Use this as a function for create gif."""
    global fig,ax,X,Y,gifData,LZ,LX,LY,LZN
    fig = plt.figure()
    ax =  plt.subplot()
    LZN = -Lz if LZn is None else LZn
    fig.colorbar(cm.ScalarMappable(cmap=cm.magma),ax=ax,boundaries=numpy.linspace(LZN,Lz,100))
    LX=Lx
    LY=Ly
    LZ=Lz
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(-Lx,Lx)
    ax.set_ylim(-Lx,Lx)
    #Create x and y data
    shape = numpy.shape(data)
    x = numpy.linspace(-Lx,Lx,shape[1])
    y = numpy.linspace(-Ly,Ly,shape[2])
    X,Y = numpy.meshgrid(x,y)
    if gif:
        gifDataRange = range(0,len(data),gmod)
        gifData = numpy.zeros((len(gifDataRange),)+numpy.shape(data)[1:])
        for i,j in enumerate(gifDataRange):
            gifData[i,:,:] = data[j,:,:]
        frames = len(gifData)
        anim = animation.FuncAnimation(fig,animateContour,frames)
        anim.save(filename.split(".")[0]+".gif",writer="imagemagick")
    else:
        ax.contourf(X,Y,data[tid],cmap=cm.magma)
        plt.savefig(filename)
   
def animateContour(i): 
    ax.contourf(X,Y,gifData[i],cmap=cm.magma,vmin=LZN, vmax=LZ)
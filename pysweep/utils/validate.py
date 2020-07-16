import h5py,numpy
#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib import animation, rc

def createSurface(data,tid,Lx,Ly,Lz,xlab="X",ylab="Y",filename="surface.pdf",gif=False,elev=45,azim=25):
    """Use this as a function for create gif."""
    global fig,ax,X,Y,gifData
    """Use this as a function for create gif."""
    fig =  plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(cm.ScalarMappable(cmap=cm.magma),ax=ax,boundaries=numpy.linspace(0,Lz,10))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(-Lx,Lx)
    ax.set_ylim(-Lx,Lx)
    ax.set_zlim(-Lz,Lz)
    #Create x and y data
    shape = numpy.shape(data)
    x = numpy.linspace(-Lx,Lx,shape[1])
    y = numpy.linspace(-Ly,Ly,shape[2])
    X,Y = numpy.meshgrid(x,y)
    if gif:
        gifData = data
        frames = len(data)
        anim = animation.FuncAnimation(fig,animateContour,frames)
        anim.save(filename.split(".")[0]+".gif",writer="imagemagick")
    else:
        ax.plot_surface(X,Y,data[tid],cmap=cm.magma)
        plt.savefig(filename)

def animateSurface(i): 
    ax.plot_surface(X,Y,gifData[i],cmap=cm.magma)

def createContourf(data,tid,Lx,Ly,Lz,xlab="X",ylab="Y",filename="contour.pdf",gif=False,gmod=1):
    """Use this as a function for create gif."""
    global fig,ax,X,Y,gifData
    fig = plt.figure()
    ax =  plt.subplot()
    fig.colorbar(cm.ScalarMappable(cmap=cm.magma),ax=ax,boundaries=numpy.linspace(0,Lz,10))
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
    ax.contourf(X,Y,gifData[i],cmap=cm.magma)
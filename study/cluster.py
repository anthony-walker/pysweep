import numpy, yaml, pysweep, itertools
import pysweep.utils.validate as validate
import matplotlib.pyplot as plt
from matplotlib import cm


def makeValidateContours():
    tsteps = 100
    npx = 48
    times = [0,259,499]
    #Euler stuf
    d = 0.1
    gamma = 1.4
    dx = 10/(640-1)
    dt = d*dx

    for k,t in enumerate(times):
        #Numerical vortex
        with h5py.File("./data/eulerOutput.hdf5","r") as f:
            data = f['data']
            validate.createContourf(data[:,0,:,:],0,5,5,1,filename="numericalVortex{}.pdf".format(k),LZn=0.4)

        #Analytical Vortex
        data = numpy.zeros((1,4,npx,npx))
        data[0,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t)
        validate.createContourf(data[:,0,:,:],0,5,5,1,filename="analyticalVortex{}.pdf".format(k),LZn=0.4)

    #Heat stuff
    alpha = 1
    dt = float(d*dx**2/alpha)
    for k,t in enumerate(times):
        #Analytical Heat Surface
        data = numpy.zeros((1,1,npx,npx))
        data[0,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,alpha=alpha)
        validate.createSurface(data[:,0,:,:],0,1,1,1,filename="analyticalHeatSurface{}.pdf".format(k))

        #Numerical Heat Surface
        with h5py.File("./data/heatOutput.hdf5","r") as f:
            data = f['data']
            validate.createSurface(data[:,0,:,:],0,1,1,1,filename="numericalHeatSurface{}.pdf".format(k))


def generateArraySizes():
    """Use this function to generate array sizes based on block sizes."""
    blocksizes = [8, 12, 16, 24, 32] #blocksizes with most options
    arraystart = 0
    divisible =[False,False]
    arraysizes = []
    for i in range(5):
        arraystart += 32*20
        while not numpy.all(divisible):
            arraystart+=blocksizes[-1]
            divisible =[arraystart%bs==0 for bs in blocksizes]
        arraysizes.append(arraystart)
    return arraysizes

def getYamlData(file,savename,equation):
    document = open(file,'r')
    yamlFile = yaml.load(document,Loader=yaml.FullLoader)
    arraySizes = []
    blockSizes = []
    shares = []
    speedup = []
    swepttime = {}
    standardtime = {}
    swepttpt = {}
    standardtpt = {}
    speedup = {}
    for key in yamlFile:
        if equation in yamlFile[key]['cpu']:
            asize = yamlFile[key]['array_shape'][2]
            bsize = yamlFile[key]['blocksize']
            cshare = yamlFile[key]['share']
            arraySizes.append(asize)
            blockSizes.append(bsize)
            shares.append(cshare)
            if yamlFile[key]['swept']:
                swepttime[(asize,bsize,cshare)] = yamlFile[key]['runtime']
                swepttpt[(asize,bsize,cshare)] = yamlFile[key]['time_per_step']
            else:
                standardtime[(asize,bsize,cshare)] = yamlFile[key]['runtime']
                standardtpt[(asize,bsize,cshare)] = yamlFile[key]['time_per_step']
        
    arraySizes = list(set(arraySizes))
    blockSizes = list(set(blockSizes))
    shares = list(set(shares))
    arraySizes.sort()
    blockSizes.sort()
    shares.sort()
    keys = list(itertools.product(arraySizes,blockSizes,shares,))

    for key in keys:
        try:
            speedup[key] = standardtime[key]/swepttime[key]
        except:
            pass

    return speedup,swepttime,standardtime,swepttpt,standardtpt,arraySizes,blockSizes,shares,keys

def getContourData(data,arraysizes,uBlocks,uShares):
    Z = np.zeros((len(uBlocks),len(uShares)))
    S,B = np.meshgrid(uShares,uBlocks)
    for i,blk in enumerate(uBlocks):
        for j,share in enumerate(uShares):
            key = (arraysize,blk,share)
            Z[i,j] = data[key]
    return B,S,Z

def performancePlot(ax,B,S,Z,minV,maxV,uBlocks,uShares,ArrSize,ccm = cm.inferno,markbest=True,markworst=True,mbc=('w','k')):
    ax.contourf(B,S*100,Z,cmap=ccm,vmin=minV,vmax=maxV)
    ax.set_title('Array Size: ${}$'.format(ArrSize))
    ax.set_ylabel('Share [%]')
    ax.set_ylabel('Block Size')
    ax.grid(color='k', linewidth=1)
    ax.set_xticks([8,16,24,32])
    ax.set_yticks([0,25,50,75,100])
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5
    if markbest:
        x,y = np.where(Z==np.amax(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)
    if markworst:
        x,y = np.where(Z==np.amin(Z))
        ax.plot(uBlocks[x],uShares[y],linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)


def makeArrayContours(data,arraysizes,blocksizes,shares,minV,maxV,cmap,cbt,fname,switch=False):
    #Make speed up figure
    fig, axes = plt.subplots(ncols=3,nrows=2)
    fig.subplots_adjust(wspace=0.2,hspace=0.3,right=0.8)
    axes = np.reshape(axes,(6,))
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    cbounds = np.linspace(minV,maxV,100)
    interval = (maxV-minV)//4 #Determine interval based on 4 points
    cbs = numpy.arange(minV,maxV,interval)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax,boundaries=cbounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbs]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbs))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title(cbt,y=1.01)
    if switch:
        mbc = ('k','w')
    else
        mbc = ('w','k')
    for i,asize in enumerate(arraySizes[1:]) #Skipping smallest size for even number of plots
        B,S,Z = getContourData(data,asize,blocksizes,shares)
        performancePlot(axes[i],B,S,Z,minV,maxV,blocksizes,shares,asize,ccm=cmap,mbc=mbc)
    plt.savefig(fig)

def getStudyContour(file,equation,appendStr=""):
    #Get data
    speedup,swepttime,standardtime,swepttpt,standardtpt,arraySizes,blockSizes,shares,keys = getYamlData(file,savename,equation)
    #Make contour
    makeArrayContours(speedup,arraySizes,blockSizes,shares,0,4,cm.inferno,'Speedup',"speedUp{}.pdf".format(equation+appendStr),switch=True)
    makeArrayContours(swepttime,arraySizes,blockSizes,shares,0,500,cm.inferno_r,'Time [s]',"sweptTime{}.pdf".format(equation+appendStr))
    makeArrayContours(standardtime,arraySizes,blockSizes,shares,0,500,cm.inferno_r,'Time [s]',"standard{}.pdf".format(equation+appendStr))
    makeArrayContours(swepttpt,arraySizes,blockSizes,shares,0,4,cm.inferno,'Time/Step',"sweptTpt{}.pdf".format(equation+appendStr),switch=True)
    makeArrayContours(swepttpt,arraySizes,blockSizes,shares,0,4,cm.inferno,'Time/Step',"standardTpt{}.pdf".format(equation+appendStr),switch=True)


if __name__ == "__main__":
    getStudyContour("./oldHardware/log.yaml",'heat',appendStr="Old")
    getStudyContour("./newHardware/log.yaml",'heat',appendStr="Old")
    getStudyContour("./oldHardware/log.yaml",'euler',appendStr="Old")
    getStudyContour("./newHardware/log.yaml",'euler',appendStr="Old")
import numpy, yaml, pysweep, itertools, h5py
import pysweep.utils.validate as validate
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib import tri

#Global data
standardSizes = numpy.asarray([160, 320, 480, 640, 800, 960, 1120])
blockSizes = numpy.asarray([8,12,16,24,32])
shares = numpy.arange(0,1.1,0.1)

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
        with h5py.File("./data/eulerOutputSwept.hdf5","r") as f:
            data = f['data']
            validate.createContourf(data[:,0,:,:],0,5,5,1,filename="./figures/numericalVortex{}.pdf".format(k),LZn=0.4)

        #Analytical Vortex
        data = numpy.zeros((1,4,npx,npx))
        data[0,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t)
        validate.createContourf(data[:,0,:,:],0,5,5,1,filename="./figuresanalyticalVortex{}.pdf".format(k),LZn=0.4)

    # #Heat stuff
    # alpha = 1
    # dt = float(d*dx**2/alpha)
    # for k,t in enumerate(times):
    #     #Analytical Heat Surface
    #     data = numpy.zeros((1,1,npx,npx))
    #     data[0,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,alpha=alpha)
    #     validate.createSurface(data[:,0,:,:],0,1,1,1,filename="analyticalHeatSurface{}.pdf".format(k))

    #     #Numerical Heat Surface
    #     with h5py.File("./data/heatOutput.hdf5","r") as f:
    #         data = f['data']
    #         validate.createSurface(data[:,0,:,:],0,1,1,1,filename="numericalHeatSurface{}.pdf".format(k))


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

def getYamlData(file,equation):
    document = open(file,'r')
    yamlFile = yaml.load(document,Loader=yaml.FullLoader)
    dataLen = len(yamlFile.keys())
    data = numpy.zeros((dataLen,6))
    
    for i,key in enumerate(yamlFile):
        if equation in yamlFile[key]['cpu']:
            asize = yamlFile[key]['array_shape'][2]
            mins = numpy.abs(standardSizes-asize)
            idx = numpy.where(numpy.amin(mins)==mins)[0][0]
            asize = standardSizes[idx]
            swept = 1 if yamlFile[key]['swept'] else 0
            data[i,0] = float(swept)
            data[i,1] = float(asize)
            data[i,2] = float(yamlFile[key]['blocksize'])
            data[i,3] = float(yamlFile[key]['share'])
            data[i,4] = float(yamlFile[key]['runtime'])
            data[i,5] = float(yamlFile[key]['time_per_step'])
    data = numpy.array(data,)
    indexes = numpy.lexsort((data[:,1],data[:,0]))
    sortedData = numpy.zeros(data.shape)
    for i,idx in enumerate(indexes):
        sortedData[i,:] = data[idx,:] 
    return sortedData,standardSizes

def getContourData(data,arraysize,uBlocks,uShares):
    
    triang = tri.Triangulation(uBlocks, uShares)
    interpolator = tri.LinearTriInterpolator(triang, data)
    S,B = numpy.meshgrid(shares,blockSizes)
    Z = interpolator(B, S)
    return B,S,Z

def performancePlot(ax,B,S,Z,minV,maxV,ArrSize,ccm=cm.inferno,markbest=True,markworst=True,mbc=('w','k')):
    ax.contourf(B,S*100,Z,cmap=ccm,vmin=minV,vmax=maxV)
    ax.set_title('Array Size: ${}$'.format(ArrSize))
    ax.set_ylabel('Share [%]')
    ax.set_xlabel('Block Size')
    ax.grid(color='k', linewidth=1)
    ax.set_xticks([8,16,24,32])
    ax.set_yticks([0,25,50,75,100])
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5

    if markbest:
        x,y = numpy.where(Z==numpy.amax(Z))
        ax.plot(blockSizes[x[0]],shares[y[0]]*100,linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)

    if markworst:
        x,y = numpy.where(Z==numpy.amin(Z))
        ax.plot(blockSizes[x[0]],shares[y[0]]*100,linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)

def makeArrayContours(data,rBlocks,rShares,cbounds,cmap,cbt,fname,switch=False):
    #Make speed up figure
    ai = lambda x: slice(int(data.shape[0]//len(standardSizes)*x),int(data.shape[0]//len(standardSizes)*(x+1)),1)
    fig, axes = plt.subplots(ncols=3,nrows=2)
    fig.subplots_adjust(wspace=0.55,hspace=0.4,right=0.75)
    axes = numpy.reshape(axes,(6,))
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar_ax,boundaries=ibounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title(cbt,y=1.01)
    if switch:
        mbc = ('k','w')
    else:
        mbc = ('w','k')
    for i,asize in enumerate(standardSizes[1:],start=1): #Skipping smallest size for even number of plots
        B,S,Z = getContourData(data[ai(i)],asize,rBlocks[ai(i)],rShares[ai(i)])
        performancePlot(axes[i-1],B,S,Z,cbounds[0],cbounds[-1],asize,ccm=cmap,mbc=mbc)
    plt.savefig(fname)

def getStudyContour(file,equation,appendStr=""):
    #Get data
    data,standardSizes = getYamlData(file,equation)
    standarddata = data[:data.shape[0]//2,:]
    sweptdata = data[data.shape[0]//2:,:]
    speedup = standarddata[:,4]/sweptdata[:,4]
    #Make contour
    makeArrayContours(speedup,sweptdata[:,2],sweptdata[:,3],numpy.arange(0.8,1.25,0.1),cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+appendStr),switch=True)
    makeArrayContours(sweptdata[:,4],sweptdata[:,2],sweptdata[:,3],numpy.arange(0,400,100),cm.inferno_r,'Clocktime [s]',"./plots/clockTimeSwept{}.pdf".format(equation+appendStr))
    makeArrayContours(standarddata[:,4],standarddata[:,2],standarddata[:,3],numpy.arange(0,400,100),cm.inferno_r,'Clocktime [s]',"./plots/clockTimeStandard{}.pdf".format(equation+appendStr))
    makeArrayContours(sweptdata[:,5],sweptdata[:,2],sweptdata[:,3],numpy.arange(0,0.7,0.1),cm.inferno_r,'Time/Step',"./plots/timePerStepSwept{}.pdf".format(equation+appendStr))
    makeArrayContours(standarddata[:,5],standarddata[:,2],standarddata[:,3],numpy.arange(0,0.7,0.1),cm.inferno_r,'Time/Step',"./plots/timePerStepStandard{}.pdf".format(equation+appendStr))


if __name__ == "__main__":
    getStudyContour("./oldHardware/log.yaml",'heat',appendStr="Old")
    # getStudyContour("./newHardware/log.yaml",'heat',appendStr="Old")
    # getStudyContour("./oldHardware/log.yaml",'euler',appendStr="Old")
    # getStudyContour("./newHardware/log.yaml",'euler',appendStr="Old")
    # makeValidateContours()
import numpy, yaml, pysweep, itertools, h5py
import pysweep.utils.validate as validate
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib import tri
from mpl_toolkits import mplot3d

#Global data
standardSizes = numpy.asarray([160, 320, 480, 640, 800, 960, 1120])
blockSizes = numpy.asarray([8,12,16,24,32])
shares = numpy.arange(0,1.1,0.1)

def validateHeat():
    npx = 48
    times = [0,100,350]
    #Numerical stuff
    d = 0.1
    dx = 10/(320-1)
    alpha = 1
    dt = float(d*dx**2/alpha)
    
    #Plotting
    elev=45
    azim=25
    fig =  plt.figure()
    axes = [fig.add_subplot(2, 3, i+1) for i in range(6)]
    fig.subplots_adjust(wspace=0.55,hspace=0.45,right=0.8)
    cbounds=numpy.linspace(-1,1,11)
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=ibounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('T(x,y,t)',y=1.01)
    for i,ax in enumerate(axes[:3]):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.yaxis.labelpad=-1
        ax.set_title('Analytical,\n$t={:0.4f}$'.format(times[i]*dt))
        data = numpy.zeros((1,1,npx,npx))
        data[0,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,t=times[i]*dt,alpha=alpha)
        validate.heatContourAx(ax,data[0,0],1,1)

    # with h5py.File("./data/heatOutputSwept.hdf5","r") as f:
    #     data = f['data']
    #     for i,ax in enumerate(axes[3:]):
    #         ax.set_xlabel("X")
    #         ax.set_ylabel("Y")
    #         ax.yaxis.labelpad=-1
    #         ax.set_title('Numerical,\n$t={:0.4f}$'.format(times[i]*dt))
    #         validate.heatContourAx(ax,data[times[i],0],1,1)

    plt.savefig("./plots/heatValidate.pdf")
    plt.close()

def validateEuler():
    npx = 48
    times = [0,259,499]
    #Euler stuf
    d = 0.1
    gamma = 1.4
    dx = 10/(640-1)
    dt = d*dx
    
    #Plotting
    elev=45
    azim=25
    fig =  plt.figure()
    axes = [fig.add_subplot(2, 3, i+1) for i in range(6)]
    fig.subplots_adjust(wspace=0.55,hspace=0.45,right=0.8)
    cbounds=numpy.linspace(0.4,1,11)
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=ibounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('$\\rho(x,y,t)$',y=1.01)
    for i,ax in enumerate(axes[:3]):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.yaxis.labelpad=-1
        ax.set_title('A:$t={:0.4f}$'.format(times[i]*dt))
        data = numpy.zeros((1,4,npx,npx))
        data[0,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t=times[i]*dt)
        validate.eulerContourAx(ax,data[0,0],1,1)

    # with h5py.File("./data/heatOutputSwept.hdf5","r") as f:
    #         data = f['data']
    #     for i,ax in enumerate(axes[3:]):
    #         ax.set_xlabel("X")
    #         ax.set_ylabel("Y")
    #         ax.yaxis.labelpad=-1
    #         ax.set_title('N:$t={:0.4f}$'.format(times[i]*dt))
    #         validate.eulerContourAx(ax,data[times[i],0],1,1)

    plt.savefig("./plots/eulerValidate.pdf")
    plt.close()
        
def generateArraySizes():
    """Use this function to generate array sizes based on block sizes."""
    blocksizes = [8, 12, 16, 20, 24, 32] #blocksizes with most options
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

def performancePlot(ax,B,S,Z,minV,maxV,ArrSize,ccm=cm.inferno,markbest=True,markworst=True,mbc=('w','k'),printer=False):
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
        Zmax = numpy.amax(Z)
        x,y = numpy.where(Z==Zmax)
        if printer:
            print(ArrSize,Zmax,x,y)
        ax.plot(blockSizes[x[0]],shares[y[0]]*100,linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)

    if markworst:
        Zmin = numpy.amin(Z)
        x,y = numpy.where(Z==Zmin)
        if printer:
            print(ArrSize,Zmin,x,y)
        ax.plot(blockSizes[x[0]],shares[y[0]]*100,linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)

def makeArrayContours(data,rBlocks,rShares,cbounds,cmap,cbt,fname,switch=False,printer=False):
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
        performancePlot(axes[i-1],B,S,Z,cbounds[0],cbounds[-1],asize,ccm=cmap,mbc=mbc,printer=printer)
    plt.savefig(fname)

def getStudyContour(file,equation,appendStr=""):
    #Get data
    data,standardSizes = getYamlData(file,equation)
    standarddata = data[:data.shape[0]//2,:]
    sweptdata = data[data.shape[0]//2:,:]
    speedup = standarddata[:,4]/sweptdata[:,4]
    print(numpy.mean(speedup))
    #Make contour
    makeArrayContours(speedup,sweptdata[:,2],sweptdata[:,3],numpy.arange(0.8,1.6,0.1),cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+appendStr),switch=True)
    makeArrayContours(sweptdata[:,4],sweptdata[:,2],sweptdata[:,3],numpy.arange(0,400,100),cm.inferno_r,'Clocktime [s]',"./plots/clockTimeSwept{}.pdf".format(equation+appendStr))
    makeArrayContours(standarddata[:,4],standarddata[:,2],standarddata[:,3],numpy.arange(0,400,100),cm.inferno_r,'Clocktime [s]',"./plots/clockTimeStandard{}.pdf".format(equation+appendStr))
    makeArrayContours(sweptdata[:,5],sweptdata[:,2],sweptdata[:,3],numpy.arange(0,0.7,0.1),cm.inferno_r,'Time/Step',"./plots/timePerStepSwept{}.pdf".format(equation+appendStr))
    makeArrayContours(standarddata[:,5],standarddata[:,2],standarddata[:,3],numpy.arange(0,0.7,0.1),cm.inferno_r,'Time/Step',"./plots/timePerStepStandard{}.pdf".format(equation+appendStr))
    return sweptdata

if __name__ == "__main__":
    print(generateArraySizes())
    # sweptOld = getStudyContour("./oldHardware/log.yaml",'heat',appendStr="Old")
    # sweptNew = getStudyContour("./newHardware/log.yaml",'heat',appendStr="New")
    # speedup=sweptOld[:,4]/sweptNew[:,4]
    # print(numpy.mean(speedup))
    # makeArrayContours(speedup,sweptNew[:,2],sweptNew[:,3],numpy.arange(0.75,1.01,0.05),cm.inferno,'Speedup',"./plots/hardwareSpeedupHeat.pdf",switch=True)

    # validateHeat()
    # validateEuler()
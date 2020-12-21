import numpy, yaml, pysweep, itertools, h5py, math
import pysweep.utils.validate as validate
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib import tri
from mpl_toolkits import mplot3d
from matplotlib.colors import Normalize

#Global data
standardSizes = numpy.asarray([160, 320, 480, 640, 800, 960, 1120])
blockSizes = numpy.asarray([8,12,16,20,24,32])
shares = numpy.arange(0,1.1,0.1)
combos = []
cutoffs = []

def calcNumberOfPts():
    npts = 0
    global combos, cutoffs
    for ss in standardSizes:
        for bs in blockSizes:
            nblks = numpy.ceil(ss/bs)
            shareset = set()
            for sh in shares:
                numerator = float('%.5f'%(sh*nblks))
                tempShare = numpy.ceil(numerator)/nblks
                # print(tempShare,nblks,sh*nblks)
                # input()
                shareset.add(tempShare)
            shareset = list(shareset)
            shareset.sort()
            for sh in shareset:
                combos.append([ss,bs,sh])
            npts+=len(shareset)
        cutoffs.append(npts)
    return npts

# print(shares.shape,blockSizes.shape,standardSizes.shape)

def validateHeat():
    #Plotting
    elev=45
    azim=25
    fig =  plt.figure()
    nrows = 2
    ncols = 3
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
    fig.subplots_adjust(wspace=0.75,hspace=0.8,right=0.8)
    #Physical Colorbar
    cbounds=numpy.linspace(0.4,1,6)
    # cbar_ax = fig.add_axes([0.89, 0.39, 0.05, 0.51]) #left bottom width height
    cbar_ax = fig.add_axes([0.89, 0.11, 0.05, 0.76]) #left bottom width height
    ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=ibounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('$\\rho(x,y,t)$',y=1.01)
    #Error Colorbar
    # cbounds=numpy.linspace(0,1e-15,3)
    # cbar_ax = fig.add_axes([0.87, 0.07, 0.05, 0.25])
    # ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    # cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds),cax=cbar_ax,boundaries=ibounds)
    # cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    # tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    # cbar_ax.set_title('Error',y=1.01)
    #Opening Results File
    with h5py.File("./data/heatOutput.hdf5","r") as f:
        data = f['data']
        times = [0,] #Times to plot
        times.append(numpy.shape(data)[0]//2)
        times.append(numpy.shape(data)[0]-1)
        #Numerical stuff
        d = 0.1
        dx = 1/(numpy.shape(data)[2]-1)
        alpha = 1
        dt = float(d*dx**2/alpha)
        for i,ax in enumerate(axes[3:6]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('N:$t={:0.2e}$'.format(times[i]*dt))
            validate.heatContourAx(ax,data[times[i],0],1,1)
        #Analytical
        npx = numpy.shape(data)[2]
        for i,ax in enumerate(axes[:3]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('A:$t={:0.2e}$'.format(times[i]*dt))
            adata = numpy.zeros((1,1,npx,npx))
            adata[0,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,t=(times[i])*dt,alpha=alpha)
            validate.heatContourAx(ax,adata[0,0],1,1)
        # #Error
        # for i,ax in enumerate(axes[6:]):
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.yaxis.labelpad=-1
        #     ax.set_title('E:$t={:0.2e}$'.format(times[i]*dt))
        #     l = numpy.linspace(0,1,npx)
        #     X,Y = numpy.meshgrid(l,l)
        #     Error = numpy.absolute(adata[0,0]-data[times[i],0])
        #     print(numpy.amax(Error))
        #     ax.contourf(X,Y,Error,cmap=cm.Reds)

    plt.savefig("./plots/heatValidate.pdf")
    plt.close()

def validateEuler(): 
    #Plotting
    elev=45
    azim=25
    fig =  plt.figure()
    nrows = 2
    ncols = 3
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
    fig.subplots_adjust(wspace=0.75,hspace=0.8,right=0.8)
    #Physical Colorbar
    cbounds=numpy.linspace(0.4,1,6)
    # cbar_ax = fig.add_axes([0.89, 0.39, 0.05, 0.51]) #left bottom width height
    cbar_ax = fig.add_axes([0.89, 0.11, 0.05, 0.76]) #left bottom width height
    ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),cax=cbar_ax,boundaries=ibounds)
    cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar_ax.set_title('$\\rho(x,y,t)$',y=1.01)
    # #Error Colorbar
    # cbounds=numpy.linspace(0,0.4,3)
    # cbar_ax = fig.add_axes([0.89, 0.07, 0.05, 0.25])
    # ibounds = numpy.linspace(cbounds[0],cbounds[-1],100)
    # cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.Reds),cax=cbar_ax,boundaries=ibounds)
    # cbar.ax.set_yticklabels([["{:0.1f}".format(i) for i in cbounds]])
    # tick_locator = ticker.MaxNLocator(nbins=len(cbounds))
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    # cbar_ax.set_title('Error',y=1.01)
    #Open Result file
    with h5py.File("./data/eulerOutput.hdf5","r") as f:
        data = f['data']
        times = [0,] #Times to plot
        times.append(numpy.shape(data)[0]//2)
        times.append(numpy.shape(data)[0]-1)
        #Euler stuff
        d = 0.1
        gamma = 1.4
        dx = 10/(640-1)
        dt = d*dx
        #Numerical
        for i,ax in enumerate(axes[3:6]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('N:$t={:0.2e}$'.format(times[i]*dt))
            validate.eulerContourAx(ax,data[times[i],0],1,1)
        #Analytical
        npx = numpy.shape(data)[2]
        for i,ax in enumerate(axes[:3]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('A:$t={:0.2e}$'.format(times[i]*dt))
            adata = numpy.zeros((1,4,npx,npx))
            adata[0,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t=(times[i])*dt)
            validate.eulerContourAx(ax,adata[0,0],1,1)
        # #Error
        # for i,ax in enumerate(axes[6:]):
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.yaxis.labelpad=-1
        #     ax.set_title('E:$t={:0.2e}$'.format(times[i]*dt))
        #     l = numpy.linspace(0,1,npx)
        #     X,Y = numpy.meshgrid(l,l)
        #     Error = numpy.absolute(adata[0,0]-data[times[i],0])
        #     # print(numpy.amax(Error))
        #     ax.contourf(X,Y,Error,cmap=cm.Reds)
    plt.savefig("./plots/eulerValidate.pdf")
    plt.close()
        
def generateArraySizes():
    """Use this function to generate array sizes based on block sizes."""
    blocksizes = [8, 12, 16, 20, 24, 32] #blocksizes with most options
    arraystart = 0
    divisible =[False,False]
    arraysizes = []
    for i in range(7):
        arraystart += 32*20
        while not numpy.all(divisible):
            arraystart+=blocksizes[-1]
            divisible =[arraystart%bs==0 for bs in blocksizes]
        arraysizes.append(arraystart)
    return arraysizes

def getYamlData(file,equation):
    document = open(file,'r')
    yamlFile = yaml.load(document,Loader=yaml.FullLoader)
    data = []
    for key in yamlFile:
        if equation in yamlFile[key]['cpu']:
            swept = 1 if yamlFile[key]['swept'] else 0
            data.append([float(swept),float(yamlFile[key]['array_shape'][2]),float(yamlFile[key]['blocksize']),float(yamlFile[key]['share']),float(yamlFile[key]['runtime']),float(yamlFile[key]['time_per_step'])])
    i = 0
    while i < len(data):
        j = i+1
        while j < len(data):
            if data[i][:4] == data[j][:4]: 
                data.pop(j)
            j+=1
        i+=1

    data = numpy.array(data,)
    indexes = numpy.lexsort((data[:,3],data[:,2],data[:,1],data[:,0]))
    sortedData = numpy.zeros(data.shape)
    for i,idx in enumerate(indexes):
        sortedData[i,:] = data[idx,:] 
    return sortedData,standardSizes

def checkForAllPts(sortedData):
    ct = 0
    for j in range(len(sortedData)):

        if list(sortedData[j,1:4])!=combos[(j+ct)%452]:
            print("{:n}, {:n}, {:n}, {:.4f}".format(*sortedData[j,:4]))
            print("{:n}, {:n}, {:.4f}".format(*combos[(j+ct)%452]))
            ct+=1
            input()

def getContourData(data,arraysize,uBlocks,uShares):
    triang = tri.Triangulation(uBlocks, uShares)
    interpolator = tri.LinearTriInterpolator(triang, data)
    S,B = numpy.meshgrid(shares,blockSizes)
    Z = interpolator(B, S)
    return B,S,Z

def performancePlot(ax,B,S,Z,minV,maxV,ArrSize,ccm=cm.inferno,markbest=True,markworst=True,mbc=('w','k'),printer=False):
    ax.contourf(B,S*100,Z,cmap=ccm,vmin=minV,vmax=maxV)
    

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

    for i in range(1,len(cutoffs),1):
        lb = cutoffs[i-1]+1
        ub = cutoffs[i]+1
        xw= numpy.where(numpy.amax(data[lb:ub])==data[lb:ub])
        xb= numpy.where(numpy.amin(data[lb:ub])==data[lb:ub])
        #Contours
        axes[i-1].tricontour(rBlocks[lb:ub],rShares[lb:ub]*100,data[lb:ub],
                 colors=('k',),linestyles=('-',),linewidths=(0.25,),vmin=cbounds[0],vmax=cbounds[-1])
        axes[i-1].tricontourf(rBlocks[lb:ub],rShares[lb:ub]*100,data[lb:ub],cmap=cmap,vmin=cbounds[0],vmax=cbounds[-1])
        #Marking best and worst
        axes[i-1].plot(rBlocks[lb:ub][xb],rShares[lb:ub][xb]*100,linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)
        axes[i-1].plot(rBlocks[lb:ub][xw],rShares[lb:ub][xw]*100,linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)

        #Labels
        axes[i-1].set_title('Array Size: ${}$'.format(standardSizes[i]))
        axes[i-1].set_ylabel('Share [%]')
        axes[i-1].set_xlabel('Block Size')
        axes[i-1].grid(color='k', linewidth=1)
        axes[i-1].set_xticks([8,16,24,32])
        axes[i-1].set_yticks([0,25,50,75,100])
        axes[i-1].yaxis.labelpad = 0.5
        axes[i-1].xaxis.labelpad = 0.5
    plt.savefig(fname)

def getContourYamlData(file,equation):
    data,standardSizes = getYamlData(file,equation)
    standarddata = data[:data.shape[0]//2,:]
    sweptdata = data[data.shape[0]//2:,:]
    speedup = standarddata[:,4]/sweptdata[:,4]
    return data,standarddata,sweptdata,standardSizes,speedup

def getDataLimits(data,npts=10):
    upperLimit = numpy.amax(data)
    lowerLimit = numpy.amin(data)
    upperLimit = math.ceil(upperLimit*10)/10
    lowerLimit = math.floor(lowerLimit*10)/10
    return numpy.linspace(lowerLimit,upperLimit,npts)
    

def getStudyContour(equation):
    #Get data
    newdata,newstandarddata,newsweptdata,newstandardSizes,newspeedup = getContourYamlData("./newHardware/log.yaml",equation)
    olddata,oldstandarddata,oldsweptdata,oldstandardSizes,oldspeedup = getContourYamlData("./oldHardware/log.yaml",equation)
    #Make contour
    #Speedup contours
    speedLimits = getDataLimits([newspeedup,oldspeedup])
    makeArrayContours(oldspeedup,oldsweptdata[:,2],oldsweptdata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+"Old"),switch=True)
    makeArrayContours(newspeedup,newsweptdata[:,2],newsweptdata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+"New"),switch=True)
    
    #Time Contours
    timeLimits = getDataLimits([oldsweptdata[:,4],newsweptdata[:,4],oldstandarddata[:,4],newstandarddata[:,4]])
    makeArrayContours(oldsweptdata[:,4],oldsweptdata[:,2],oldsweptdata[:,3],timeLimits,cm.inferno_r,'Clocktime [s]',"./plots/clockTimeSwept{}.pdf".format(equation+"Old"))
    makeArrayContours(newsweptdata[:,4],newsweptdata[:,2],newsweptdata[:,3],timeLimits,cm.inferno_r,'Clocktime [s]',"./plots/clockTimeSwept{}.pdf".format(equation+"New"))
    makeArrayContours(oldstandarddata[:,4],oldstandarddata[:,2],oldstandarddata[:,3],timeLimits,cm.inferno_r,'Clocktime [s]',"./plots/clockTimeStandard{}.pdf".format(equation+"Old"))
    makeArrayContours(newstandarddata[:,4],newstandarddata[:,2],newstandarddata[:,3],timeLimits,cm.inferno_r,'Clocktime [s]',"./plots/clockTimeStandard{}.pdf".format(equation+"New"))

    #Time per timestep contours
    timePerStepLimits = getDataLimits([oldsweptdata[:,5],newsweptdata[:,5],oldstandarddata[:,5],newstandarddata[:,5]])
    makeArrayContours(oldsweptdata[:,5],oldsweptdata[:,2],oldsweptdata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepSwept{}.pdf".format(equation+"Old"))
    makeArrayContours(newsweptdata[:,5],newsweptdata[:,2],newsweptdata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepSwept{}.pdf".format(equation+"New"))
    makeArrayContours(oldstandarddata[:,5],oldstandarddata[:,2],oldstandarddata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepStandard{}.pdf".format(equation+"Old"))
    makeArrayContours(newstandarddata[:,5],newstandarddata[:,2],newstandarddata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepStandard{}.pdf".format(equation+"New"))
    #3D plot
    # ThreeDimPlot(oldsweptdata[:,2],100*oldsweptdata[:,3],oldspeedup,speedLimits)
    return oldsweptdata,newsweptdata


def ThreeDimPlot(x,y,z,lims):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x,y,z,cmap=cm.inferno)
    ax.set_xticks([8,16,24,32])
    ax.set_yticks([0,25,50,75,100])
    ax.set_zlim([lims[0],lims[-1]])
    ax.yaxis.labelpad = 0.5
    ax.xaxis.labelpad = 0.5
    plt.savefig("./plots/weakScalability.pdf")
    # plt.show()

def ScalabilityPlots():
    data,standardSizes = getYamlData("./scalability/scalability.yaml","euler")
    dl = len(data)
    sweptdata = data[dl//2:]
    standarddata = data[:dl//2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Clocktime [s]")
    ax.set_xlabel("Array Size")
    # ax.set_title("Weak Scalability")
    l1 = ax.plot(sweptdata[:,1],sweptdata[:,4],marker="o",color='b')
    l2 = ax.plot(standarddata[:,1],standarddata[:,4],marker="o",color="r")
    ax.legend(["Swept","Standard"])
    plt.show()

if __name__ == "__main__":
    calcNumberOfPts()
    
    # Produce contour plots
    sweptEuler = getStudyContour('heat')
    sweptEuler = getStudyContour('euler')

    # Scalability
    ScalabilityPlots()

    # Produce physical plots
    validateHeat()
    validateEuler()
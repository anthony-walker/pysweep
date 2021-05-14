import numpy, yaml, itertools, h5py, math
import pysweep.utils.validate as validate
import pysweep.tests as tests
import pysweep
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib import tri
from mpl_toolkits import mplot3d
from matplotlib.colors import Normalize
import mpi4py.MPI as MPI
import os
from scipy.stats import linregress

#Global data
standardSizes = numpy.asarray([160, 320, 480, 640, 800, 960, 1120])
blockSizes = numpy.asarray([8,12,16,20,24,32])
best = []
worst = []
shares = numpy.arange(0,1.1,0.1)
combos = []
cutoffs = [0,]

def switchColorScheme(printParams=False):
    matplotlib.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white","grid.color":"white","savefig.facecolor":"#333333","savefig.edgecolor":"#333333","axes.facecolor":"#333333"})
    if printParams:
        for k in matplotlib.rcParams:
            if "color" in k:
                print(k)

def printArray(a):
    for row in a:
        for item in row:
            print("{:8.3f}".format(col), end=" ")
        print("")

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

def validateClusterEuler():
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
    cbar_ax.set_title('$\\rho(x,y,t)$\n$[kg/m^3]$',y=1.01)
    with h5py.File("./data/eulerOutput.hdf5","r") as f:
        data = f['data']
        times = [0,] #Times to plot
        times.append(numpy.shape(data)[0]//2)
        times.append(numpy.shape(data)[0]-1)
        #Euler stuff
        d = 0.1
        gamma = 1.4
        dx = 10/(data.shape[-1]-1)
        dt = d*dx
        #Numerical
        for i,ax in enumerate(axes[3:6]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('N:$t={:0.2f}$ [s]'.format(times[i]*dt))
            validate.eulerContourAx(ax,data[times[i],0],1,1)
            ax.grid('on',color="k")
        #Analytical
        npx = numpy.shape(data)[2]
        for i,ax in enumerate(axes[:3]):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.yaxis.labelpad=-1
            ax.set_title('A:$t={:0.2f}$ [s]'.format(times[i]*dt))
            adata = numpy.zeros((1,4,npx,npx))
            adata[0,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t=(times[i])*dt)
            validate.eulerContourAx(ax,adata[0,0],1,1)
            ax.grid('on',color="k")
    plt.savefig("./plots/eulerValidate.pdf",bbox_inches='tight')
    plt.close('all')

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

def makeArrayContours(data,rBlocks,rShares,cbounds,cmap,cbt,fname,switch=False,printer=False,record=False):
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
    for i in range(0,len(cutoffs)-2,1):
        lb = cutoffs[i+1]
        ub = cutoffs[i+2]
        xw= numpy.where(numpy.amax(data[lb:ub])==data[lb:ub])
        xb= numpy.where(numpy.amin(data[lb:ub])==data[lb:ub])
        #Contours
        axes[i].tricontour(rBlocks[lb:ub],rShares[lb:ub]*100,data[lb:ub],
                 colors=('k',),linestyles=('-',),linewidths=(0.25,),vmin=cbounds[0],vmax=cbounds[-1])
        axes[i].tricontourf(rBlocks[lb:ub],rShares[lb:ub]*100,data[lb:ub],cmap=cmap,vmin=cbounds[0],vmax=cbounds[-1])
        #Marking best and worst
        axes[i].plot(rBlocks[lb:ub][xb],rShares[lb:ub][xb]*100,linestyle=None,marker='o',markerfacecolor=mbc[0],markeredgecolor=mbc[1],markersize=6)
        axes[i].plot(rBlocks[lb:ub][xw],rShares[lb:ub][xw]*100,linestyle=None,marker='o',markerfacecolor=mbc[1],markeredgecolor=mbc[0],markersize=6)
        if record:
            worst.append((rBlocks[lb:ub][xb],rShares[lb:ub][xb]))
            best.append((rBlocks[lb:ub][xw],rShares[lb:ub][xw]))
        #Labels
        axes[i].set_title('Array Size: ${}$'.format(standardSizes[i+1]))
        axes[i].set_ylabel('Share [%]')
        axes[i].set_xlabel('Block Size')
        axes[i].grid(color='k', linewidth=1)
        axes[i].set_xticks([8,16,24,32])
        axes[i].set_yticks([0,25,50,75,100])
        axes[i].yaxis.labelpad = 0.5
        axes[i].xaxis.labelpad = 0.5
    plt.savefig(fname,bbox_inches='tight')
    plt.close('all')

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
    makeArrayContours(oldspeedup,oldsweptdata[:,2],oldsweptdata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+"Old"),switch=True,record=True)
    makeArrayContours(newspeedup,newsweptdata[:,2],newsweptdata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/speedUp{}.pdf".format(equation+"New"),switch=True,record=True)
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
    
    #Speedup based on time per timestep
    new_tpt_speedup = newstandarddata[:,5]/newsweptdata[:,5]
    old_tpt_speedup = oldstandarddata[:,5]/oldsweptdata[:,5]
    tptSpeedLimits = getDataLimits([new_tpt_speedup,old_tpt_speedup])
    makeArrayContours(new_tpt_speedup,newsweptdata[:,2],newsweptdata[:,3],tptSpeedLimits,cm.inferno,'Speedup',"./plots/SpeedupTPT{}.pdf".format(equation+"New"))
    makeArrayContours(old_tpt_speedup,oldsweptdata[:,2],oldsweptdata[:,3],tptSpeedLimits,cm.inferno,'Speedup',"./plots/SpeedupTPT{}.pdf".format(equation+"Old"))
    
    # makeArrayContours(oldstandarddata[:,5],oldstandarddata[:,2],oldstandarddata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepStandard{}.pdf".format(equation+"Old"))
    # makeArrayContours(newstandarddata[:,5],newstandarddata[:,2],newstandarddata[:,3],timePerStepLimits,cm.inferno_r,'Time/Step [s]',"./plots/timePerStepStandard{}.pdf".format(equation+"New"))
    

    #Hardware Speedup
    hardwareSpeedUp = oldsweptdata[:,5]/newsweptdata[:,5]
    hardwareSpeedUpStand = oldstandarddata[:,5]/newstandarddata[:,5]
    # print("Hardware Speed Up Average {}:{:0.2f}".format(equation,numpy.mean(hardwareSpeedUp)))
    # print("Hardware Speed Up Min {}:{:0.2f}".format(equation,numpy.amin(hardwareSpeedUp)))
    # print("Hardware Speed Up Max {}:{:0.2f}\n".format(equation,numpy.amax(hardwareSpeedUp)))
    speedLimits = getDataLimits([hardwareSpeedUp,])
    # if equation == "heat":
    #     speedLimits = numpy.linspace(1,2,11)
    makeArrayContours(hardwareSpeedUp,oldsweptdata[:,2],oldsweptdata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/hardwareSpeedUp{}.pdf".format(equation),switch=True)
    makeArrayContours(hardwareSpeedUpStand,oldstandarddata[:,2],oldstandarddata[:,3],speedLimits,cm.inferno,'Speedup',"./plots/hardwareStandardSpeedUp{}.pdf".format(equation),switch=True)

    # print("New Speed Up Average {}:{:0.2f}".format(equation,numpy.mean(newspeedup)))
    # print("Old Speed Up Average {}:{:0.2f}".format(equation,numpy.mean(oldspeedup)))
    print("New Speed Up Min,Max {}:{:0.2f},{:0.2f}".format(equation,numpy.amin(new_tpt_speedup),numpy.amax(new_tpt_speedup)))
    print("Old Speed Up Min,Max {}:{:0.2f},{:0.2f}".format(equation,numpy.amin(old_tpt_speedup),numpy.amax(old_tpt_speedup)))
    

    # 3D plot
    ThreeDimPlot(oldsweptdata[:,2],100*oldsweptdata[:,3],oldspeedup,speedLimits)
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
    # plt.show()

def lineregressSlope(d1,d2,name):
    slope, intercept, r_value, p_value, std_err = linregress(d1, d2)
    print(name,slope)
    return slope

def ScalabilityPlots():
    plt.close()
    data,standardSizes = getYamlData("./scalability/scalability.yaml","euler")
    dl = len(data)
    eulersweptdata = data[dl//2:]
    eulerstandarddata = data[:dl//2]
    data,standardSizes = getYamlData("./scalability/scalability.yaml","heat")
    dl = len(data)
    heatsweptdata = data[dl//2:]
    heatstandarddata = data[:dl//2]
    lsty = "--"
    #Euler
    fig, ax = plt.subplots(ncols=1,nrows=1)
    ax.set_ylabel("Time per Timestep [s]")
    ax.set_xlabel("Spatial Points")
    # ax.set_title("Compressible Euler Equations")
    spacedata = eulersweptdata[:,1]*eulersweptdata[:,1] #2580*numpy.arange(1,5,1)#
    # ax.axis([10**6,10**7, 10, 30])
    l1 = ax.loglog(spacedata,eulersweptdata[:,5],marker="o",color='#d95f02')
    l2 = ax.loglog(spacedata,eulerstandarddata[:,5],marker="s",color="#7570b3",linestyle=lsty)
    lrgsw = lineregressSlope(spacedata,eulersweptdata[:,5],"Swept Euler ")
    lrgst = lineregressSlope(spacedata,eulerstandarddata[:,5],"Standard Euler ")
    # ax.annotate('m={:.2f}'.format(lrgsw), xy=(eulersweptdata[-2,1],eulersweptdata[-2,5]), xytext=(eulersweptdata[-2,1]-1000, eulersweptdata[-2,5]+500),arrowprops=dict(facecolor='black', arrowstyle="simple"))
    # ax.annotate('m={:.2f}'.format(lrgst), xy=(eulerstandarddata[-2,1],eulerstandarddata[-2,4]), xytext=(eulerstandarddata[-2,1]-500, eulerstandarddata[-2,4]+1000),arrowprops=dict(facecolor='black', arrowstyle="simple"))
    leg = ax.legend(["Swept","Standard"])
    plt.savefig("./plots/weakScalabilityEuler.pdf",bbox_inches='tight')
    plt.close('all')

    #Heat
    fig, ax = plt.subplots(ncols=1,nrows=1)
    # ax.axis([10**6,10**7, 3* 10**-1, 10**0])
    ax.yaxis.set_label_coords(-0.135,0.5)
    ax.set_ylabel("Time per Timestep [s]")
    ax.set_xlabel("Spatial Points")
    # ax.set_title("Heat Diffusion Equation")
    l1 = ax.loglog(spacedata,heatsweptdata[:,5],marker="o",color='#d95f02')
    l2 = ax.loglog(spacedata,heatstandarddata[:,5],marker="s",color="#7570b3",linestyle=lsty)
    lrgsw = lineregressSlope(spacedata,heatsweptdata[:,5],"Swept Heat ")
    lrgst = lineregressSlope(spacedata,heatstandarddata[:,5],"Standard Heat ")
    # ax.annotate('m={:.2f}'.format(lrgsw), xy=(heatsweptdata[-2,1],heatsweptdata[-2,5]), xytext=(heatsweptdata[-2,1]-1000, heatsweptdata[-2,5]),arrowprops=dict(facecolor='black', arrowstyle="simple"))
    # ax.annotate('m={:.2f}'.format(lrgst), xy=(heatstandarddata[-2,1],heatstandarddata[-2,4]), xytext=(heatstandarddata[-2,1], heatstandarddata[-2,4]-30),arrowprops=dict(facecolor='black', arrowstyle="simple"))
    leg = ax.legend(["Swept","Standard"])
    plt.savefig("./plots/weakScalabilityHeat.pdf",bbox_inches='tight')
    plt.close('all')
    # plt.show()

def getOccurences(vals,targets):
    L = len(vals)
    return [vals.count(i)/L*100 for i in targets]

def modePlots():
    global best, worst
    best = [(i[0],round(j[0]*10,0)/10) for i, j in best]
    worst = [(i[0],round(j[0]*10,0)/10) for i, j in worst]

    bBlocks,bShares = zip(*best)
    wBlocks,wShares = zip(*worst)
    bestBlockOccurences = getOccurences(bBlocks,blockSizes) #[:len(bBlocks//2)]
    worstBlockOccurences = getOccurences(wBlocks,blockSizes)
    bestShareOccurences = getOccurences(bShares,shares)
    worstShareOccurences = getOccurences(wShares,shares)
    fig =  plt.figure()
    nrows = 1
    ncols = 2
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
    fig.subplots_adjust(wspace=0.3)

    ax1,ax2 = axes

    ax1.set_ylabel("Percentage Of Occurences[%]")
    ax1.set_xlabel("Block Size")
    ax1.plot(blockSizes,bestBlockOccurences,marker="o",color='#d95f02')
    ax1.plot(blockSizes,worstBlockOccurences,marker="o",color="#7570b3")
    leg = ax1.legend(["Best","Worst"])

    ax2.set_ylabel("Percentage Of Occurences[%]")
    ax2.set_xlabel("Share")
    ax2.plot(shares,bestShareOccurences,marker="o",color='#d95f02')
    ax2.plot(shares,worstShareOccurences,marker="o",color="#7570b3")
    leg = ax2.legend(["Best","Worst"])
    plt.savefig("./plots/caseModes.pdf",bbox_inches='tight')

def validateHeat(adata,ndata,times):
    #Plotting
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
    cbar_ax.set_title('$T(x,y,t) [K]$',y=1.01)
    #Numerical stuff
    for i,ax in enumerate(axes[3:6]):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.yaxis.labelpad=-1
        ax.set_title('N:$t={:0.2f}$ [s]'.format(times[i]))
        validate.heatContourAx(ax,ndata[i],1,1)

    #Analytical
    npx = numpy.shape(adata)[-1]
    for i,ax in enumerate(axes[:3]):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.yaxis.labelpad=-1
        ax.set_title('N:$t={:0.2f}$ [s]'.format(times[i]))
        validate.heatContourAx(ax,adata[i],1,1)
    plt.savefig("./plots/heatValidate.pdf",bbox_inches='tight')
    plt.close('all')

def PresentationHDE():
    #MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  #current rank

    if rank==0:
        validate.switchColorScheme() #Change color scheme

    npx = 120
    tsteps = 2000
    alpha = 0.01
    tf,dt = pysweep.equations.heat.getFinalTime(npx,tsteps,alpha=alpha)
    gifmod=100
    adata = numpy.zeros((3,npx,npx))
    ndata = numpy.zeros((3,npx,npx))
    ndata = numpy.zeros((3,npx,npx))
    times = numpy.zeros((3,))
    #create conditions
    pysweep.equations.heat.createInitialConditions(npx,npx)

    #Numerical
    tests.testHeatForwardEuler(npx,timesteps=tsteps,runTest=False)

    #Numerical plots
    with h5py.File("testingHeatFE.hdf5","r",driver="mpio",comm=comm) as f:
        data = f['data']
        finalsteps = data.shape[0]
        if rank == 0:
            validate.createSurface(data[:,0,:,:],0,1,1,1,gif=True,gmod=gifmod,filename="/home/anthony-walker/nrg-swept-project/pysweep-git/study/plots/numericalHeatSurface.pdf")
            validate.createContourf(data[:,0,:,:],0,1,1,1,gif=True,gmod=gifmod,filename="/home/anthony-walker/nrg-swept-project/pysweep-git/study/plots/numericalHeatContour.pdf")
            for i,ts in enumerate([finalsteps//10,finalsteps//2,finalsteps-1]):
                ndata[i,:,:] = data[ts,0,:,:]
        comm.Barrier()

    #Analytical solution
    if rank == 0:
        points = [(int(i),k) for i,k in enumerate(numpy.linspace(0,tf,tsteps))]
        split_points = numpy.array_split(points,comm.Get_size())
    else:
        split_points = None
    split_points = comm.scatter(split_points)

    #Analytical plots
    with h5py.File('temp.hdf5', 'w', driver="mpio",comm=comm) as f:
        data = f.create_dataset("data", (tsteps,1,npx,npx))
        for i,t in split_points:
            T,x,y = pysweep.equations.heat.analytical(npx,npx,t,alpha=0.01)
            data[int(i),0,:,:] = T[0,:,:]
        if rank==0:
            validate.createSurface(data[:,0,:,:],0,1,1,1,gif=True,gmod=gifmod,filename="/home/anthony-walker/nrg-swept-project/pysweep-git/study/plots/analyticalHeatSurface.pdf")
            validate.createContourf(data[:,0,:,:],0,1,1,1,gif=True,gmod=gifmod,filename="/home/anthony-walker/nrg-swept-project/pysweep-git/study/plots/analyticalHeatContour.pdf")
            for i,ts in enumerate([finalsteps//10,finalsteps//2,finalsteps-1]):
                adata[i,:,:] = data[ts,0,:,:]
                times[i] =  dt*ts
    comm.Barrier()

    #Clean up and other plots
    if rank == 0:
        validateHeat(adata,ndata,times)
        os.system("rm temp.hdf5 heatConditions.hdf5 testingHeatFE.hdf5")

def generateOtherEulerGIFs():
    npx = 48
    tsteps = 100
    gmod = 5
    mul=3
    data = numpy.zeros((int(mul*tsteps),4,npx,npx))
    sdata = numpy.zeros((tsteps,4,npx,npx))
    for i,t in enumerate(numpy.linspace(0,mul,mul*tsteps)):
        data[i,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t)
    for i,t in enumerate(numpy.linspace(0,1,tsteps)):
        sdata[i,:,:,:] = pysweep.equations.euler.getPeriodicShock(npx,t)
    validate.createContourf(data[:,0,:,:],0,5,5,1,gif=True,gmod=mul*gmod,filename="analyticalVortex.pdf",LZn=0.4)
    validate.createSurface(sdata[:,0,:,:],0,1,1,1,gif=True,gmod=gmod,filename="analyticalShock.pdf")

def generateEuler(comm,npx,tsteps,gifmod):
    #Numerical
    rank = comm.Get_rank()  #current rank
    pysweep.equations.euler.createInitialConditions(npx,npx) #create initial conditions
    tests.testEulerVortex(npx,timesteps=tsteps,runTest=False) #Run test

    with h5py.File('testingVortex.hdf5', 'r', driver="mpio",comm=comm) as f:
        data = f['data']
        finalsteps = data.shape[0]
        if rank == 0:
            validate.createContourf(data[:,0,:,:],0,5,5,1,gif=True,gmod=gifmod,filename="/home/anthony-walker/nrg-swept-project/pysweep-git/study/plots/numericalVortex{:d}.pdf".format(npx),LZn=0.4,gridBool=True)
    comm.Barrier()

def PresentationEuler():
    """Use this function to produce presentation data"""
    #MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  #current rank

    npx1 = 120
    tsteps = 1000
    gamma=1.4
    d=0.1
    tf,dt = pysweep.equations.euler.getFinalTime(npx1,tsteps,gamma=gamma)

    if rank==0:
        validate.switchColorScheme() #Change color scheme
    sizes = [npx1,360,720]
    for npx in sizes:
        dx = 10/npx
        dt = d*dx
        tsteps = tf//dt
        gifmod = int((tsteps//20))
        generateEuler(comm,npx,tsteps,gifmod)


    #Clean up and other plots
    if rank == 0:
        os.system("eulerConditions.hdf5 testingVortex.hdf5")

if __name__ == "__main__":
    #Find number of points there should be
    calcNumberOfPts()

    #Presentation bool
    pres = False
    #Switch colors to match presentation
    if pres:
        switchColorScheme()

    # Produce contour plots
    sweptEuler = getStudyContour('heat')
    sweptEuler = getStudyContour('euler')

    # Scalability
    ScalabilityPlots()

    # # validateClusterEuler()

    # # Produce presentation figures
    # if pres:
    #     PresentationEuler()
    #     PresentationHDE()
    #     generateOtherEulerGIFs()
    # # mode plots of best and worst cases
    # modePlots()

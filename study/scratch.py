import pysweep,numpy,sys,h5py
import pysweep.tests as tests
import pysweep.utils.validate as validate
import mpi4py
mpi4py.rc.recv_mprobe = False
import mpi4py.MPI as MPI

#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  #current rank

tsteps = 100
npx = 48
vortex = True
shock = True
heatSurface = True
heatContour = False
main = rank == 0

if vortex:
    tests.testEulerVortex() #run euler vortex test
    if main:
        data = numpy.zeros((tsteps,4,npx,npx))
        for i,t in enumerate(numpy.linspace(0,1,tsteps)):
            data[i,:,:,:] = pysweep.equations.euler.getAnalyticalArray(npx,npx,t)
        validate.createContourf(data[:,0,:,:],0,5,5,1,gif=True,gmod=5,filename="analyticalVortex.pdf",LZn=0.4)

        with h5py.File("testingVortex.hdf5","r") as f:
            data = f['data']
            validate.createContourf(data[:,0,:,:],0,5,5,1,gif=True,gmod=20,filename="numericalVortex.pdf",LZn=0.4)
elif shock:
    tests.testEulerShock() #run euler shock test
    if main:
        data = numpy.zeros((tsteps,4,npx,npx))
        for i,t in enumerate(numpy.linspace(0,1,tsteps)):
            data[i,:,:,:] = pysweep.equations.euler.getPeriodicShock(npx,t)
        validate.createSurface(data[:,0,:,:],0,1,1,1,gif=True,gmod=5,filename="analyticalVortex.pdf",LZn=0)

        with h5py.File("testingShock.hdf5","r") as f:
            data = f['data']
            validate.createContourf(data[:,0,:,:],0,1,1,1,gif=True,gmod=10,filename="numericalShock.pdf",LZn=0)
elif heatSurface or heatContour:
    tests.testHeatForwardEuler() #run heat FE
    if main:
        if heatSurface:
            data = numpy.zeros((tsteps,1,npx,npx))
            for i,t in enumerate(numpy.linspace(0,1,tsteps)):
                data[i,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,t)
            validate.createSurface(data[:,0,:,:],0,1,1,1,gif=True,gmod=2,filename="analyticalHeatSurface.pdf")

            with h5py.File("testingHeatFE.hdf5","r") as f:
                data = f['data']
                validate.createSurface(data[:,0,:,:],0,1,1,1,gif=True,gmod=100,filename="numericalHeatSurface.pdf")

        if heatContour:
            data = numpy.zeros((tsteps,1,npx,npx))
            for i,t in enumerate(numpy.linspace(0,1,tsteps)):
                data[i,:,:,:],x,y = pysweep.equations.heat.analytical(npx,npx,t)
            validate.createContourf(data[:,0,:,:],0,1,1,1,gif=True,gmod=2,filename="analyticalHeatContour.pdf",LZn=-1)

            with h5py.File("testingHeatFE.hdf5","r") as f:
                data = f['data']
                validate.createContourf(data[:,0,:,:],0,1,1,1,gif=True,gmod=100,filename="numericalHeatContour.pdf")
               
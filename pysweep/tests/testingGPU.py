import numpy,sys
#Pysweep stuff
import pysweep.equations.euler as euler
import pysweep.tests as tests
import pysweep.core.kernel as kt
import pysweep.core.io as io
#Cuda stuff
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.autoinit
except Exception as e:
    pass

def writeOut(array):
    for row in array:
        sys.stdout.write("[")
        for value in row:
            sys.stdout.write("{:0.5f},".format(abs(value)))
        sys.stdout.write("]\n")

def write1D(array):
    sys.stdout.write("[")
    for value in array:
        sys.stdout.write("{:0.6f},".format(abs(value)))
    sys.stdout.write("]\n")

def updateBCX(state,time):
    state[time,:,:2,:] = state[time-1,:,:2,:]
    state[time,:,-2:,:] = state[time-1,:,-2:,:]

def updateBCY(state,time):
    state[time,:,:,:2] = state[time-1,:,:,:2]
    state[time,:,:,-2:] = state[time-1,:,:,-2:]

def testBoundaryUpdate(state,ops):
    """Use this function to update BC's during testing."""
    lBC = (1.0,0,0,2.5)
    rBC = (0.125,0,0,0.03125)
    #Left boundary
    for i in range(ops):
        for j in range(state.shape[2]):
            state[:,i,j] = lBC[:]
    #right Boundary
    for i in range(state.shape[1]-ops,state.shape[1]):
        for j in range(state.shape[2]):
            state[:,i,j] = rBC[:]
    #bottom boundary
    for i in range(state.shape[2]):
        for k,j in enumerate(range(ops),start=state.shape[2]-2*ops):
            state[:,i,j] = state[:,i,k]
    #top boundary
    for i in range(state.shape[2]):
        for k,j in enumerate(range(state.shape[2]-ops,state.shape[2],1),start=ops):
            state[:,i,j] = state[:,i,k]

def runCPU(stateCPU,timesteps,ops=2):
    for gts,i in enumerate(range(timesteps),start=1):
        #Steps
        print("---------{}---------".format(i))
        euler.step(stateCPU,iidx,i,gts)
        #Update BCs
        testBoundaryUpdate(stateCPU[i+1],ops)
        # writeOut(stateCPU[i+1,0])
        input()

def runGPU(stateGPU,timesteps,ops=2):
    for gts,i in enumerate(range(timesteps),start=1):
        print("---------{}---------".format(i))
        #Steps
        kt.step(stateGPU,i,gts)
        #Update BCs
        testBoundaryUpdate(stateGPU[i+1],ops)
        # writeOut(stateGPU[i+1,0])
        input()

if __name__ == "__main__":
    sourceMod = io.buildGPUSource("/home/anthony-walker/nrg-swept-project/pysweep-git/pysweep/equations/euler.cu")

    size = 10
    dt = 0.0001
    dx = 1/(size-1)
    euler.set_globals(0,0.1,dt,dx,dx,1.4,source_mod=sourceMod)
    ops = 2
    timesteps = 1000
    ic = euler.getShock(size,0,direc=False)
    stateGPU = numpy.zeros((timesteps+1,ic.shape[0],size+int(2*ops),size+int(2*ops)))
    stateGPU[0,:,ops:-ops,ops:-ops] = ic[:,:,:]
    testBoundaryUpdate(stateGPU[0],ops)
    stateCPU = numpy.copy(stateGPU)
    iidx = [(i+ops,j+ops) for i,j in numpy.ndindex(ic.shape[1:])]

    #GPU setup stuff
    kt.setSweptConstants(stateGPU.shape,sourceMod)
    kt.cudaArrayMalloc(stateGPU,sourceMod)


    # runCPU(stateCPU,timesteps,ops)
    # runGPU(stateGPU,timesteps,ops)

    for gts,i in enumerate(range(timesteps),start=1):
        #Steps
        euler.step(stateCPU,iidx,i,gts)
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        kt.step(stateGPU,i,gts)
        #Update BCs
        testBoundaryUpdate(stateGPU[i+1],ops)
        testBoundaryUpdate(stateCPU[i+1],ops)
        diff = stateGPU[i+1]-stateCPU[i+1]
        maxdiff = numpy.amax(diff)
        if maxdiff>=1e-16:
            print('___________________________')
            print(i,maxdiff)
            # writeOut(stateCPU[i+1,0])
            # print('___________________________')
            # writeOut(stateGPU[i+1,0])
            # print('___________________________')
            # input()
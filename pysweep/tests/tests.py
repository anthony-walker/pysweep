
import pysweep,numpy,sys,os,h5py

path = os.path.dirname(os.path.abspath(__file__))


def testExample(share=0.5,npx=768,npy=768):
    filename = pysweep.equations.example.createInitialConditions(1,npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"example.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver.share = share
    testSolver.simulation=True
    testSolver()

    if testSolver.clusterMasterBool:
        with h5py.File(testSolver.output,"r") as f:
            data = f["data"]
            for i in range(1,testSolver.arrayShape[0]+1):
                try:
                    assert numpy.all(data[i-1,0,:,:]==i)
                except Exception as e:
                    print("Simulation failed on index: {}.".format(i))
                    print(data[i-1,0,:,:])
                    input()

def testHeat(share=0.5,npx=16,npy=16):
    filename = pysweep.equations.heat.createInitialConditions(npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"heat.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver()

    # if testSolver.clusterMasterBool:
    #     with h5py.File(testSolver.output,"r") as f:
    #         data = f["data"]
    #         for i in range(1,testSolver.arrayShape[0]+1):
    #             try:
    #                 assert numpy.all(data[i-1,0,:,:]==i)
    #             except Exception as e:
    #                 print("Simulation failed on index: {}.".format(i))
    #                 print(data[i-1,0,:,:])


if __name__ == "__main__":
    pass
    # MPI.COMM_SELF.Spawn(sys.executable, args=["from pysweep.tests import testExample; testExample()"], maxprocs=2)

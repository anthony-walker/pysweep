
import pysweep,numpy

def testExample():
    npx = 32
    arr0 = numpy.zeros((4,npx,npx))
    yfile = "test.yaml"
    newSolver = pysweep.Solver(arr0,yfile)
    newSolver()
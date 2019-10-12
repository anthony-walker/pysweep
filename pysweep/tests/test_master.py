#Programmer: Anthony Walker
from test_decomp import *
from test_pysweep import *
from test_figures import *
#Use this file to implement multiple functions from test files
if __name__ == "__main__":
    tf = 8
    npx=npy=40
    aff = 0.75
    X=Y=10
    Fo = 0.24
    dt = Fo*(X/npx)**2
    alpha = 5
    dt = Fo*(X/npx)**2/alpha

    test_sweep_hde()
    test_decomp_hde()
    comp_gif()

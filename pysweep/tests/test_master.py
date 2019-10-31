#Programmer: Anthony Walker
import sys, os
sys.path.insert(0, './pysweep')
import test_decomp
import test_pysweep
import test_figures

#Use this file to implement multiple functions from test files
if __name__ == "__main__":
    tf = 8
    npx=npy=60
    BS = 12
    aff = 0.5
    X=Y=10
    Fo = 0.24
    dt = Fo*(X/npx)**2
    alpha = 5
    dt = Fo*(X/npx)**2/alpha
    args1=(tf,npx,aff,X,Fo,alpha,BS,3)
    args2=(tf,npx,aff,X,Fo,alpha,BS,4)
    print(args1)
    # test_pysweep.test_dsweep_hde(args1)
    # test_decomp.test_decomp_hde(args2)
    # test_figures.comp_gif()

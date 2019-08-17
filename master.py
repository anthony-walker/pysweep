#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
from src.sweep.sweep import *
from src.analytical import *
from src.equations import *
from src.decomp import *
import multiprocessing as mp
from notipy.notipy import NotiPy

#MPI Info
comm = MPI.COMM_WORLD
master_rank = 0 #master rank
rank = comm.Get_rank()  #current rank


#Calling analytical solution
def analytical():
    """Use this funciton to solve the analytical euler vortex."""
    # create_vortex_data(cvics,X,Y,npx,npy,times=(0,0.1))
    pass

def create_block_sizes():
    """Use this function to create arguements for the two codes."""
    #Block_sizes
    bss = list()
    for i in range(3,6,1):
        cbs = (int(2**i),int(2**i),1)
        bss.append(cbs)
    return bss


def test(args):
    #Properties
    gamma = 1.4

    #Analytical properties
    cvics = vics()
    cvics.Shu(gamma)
    initial_args = cvics.get_args()
    X = cvics.L
    Y = cvics.L
    #Dimensions and steps
    npx = 20
    npy = 20
    dx = X/npx
    dy = Y/npy

    #Time testing arguments
    t0 = 0
    t_b = 0.3
    dt = 0.1
    targs = (t0,t_b,dt)

    # Creating initial vortex from analytical code
    initial_vortex = vortex(cvics,X,Y,npx,npy,times=(0,))
    initial_vortex = np.swapaxes(initial_vortex,0,2)
    initial_vortex = np.swapaxes(initial_vortex,1,3)[0]

    #GPU Arguments
    kernel = "/home/walkanth/pysweep/src/equations/euler.h"
    cpu_source = "/home/walkanth/pysweep/src/equations/euler.py"
    ops = 2 #number of atomic operations
    #File args
    swept_name = "./results/swept"
    decomp_name = "./results/decomp"
    #Changing arguments
    affinities = np.linspace(1/2,1,mp.cpu_count()/2)
    block_sizes = create_block_sizes()
    if rank == master_rank:
        f =  open("./results/time_data.txt",'w')
    # #Swept results
    # for i,bs in enumerate(block_sizes):
    #     for j,aff in enumerate(affinities):
    #         fname = swept_name+"_"+str(i)+"_"+str(j)
    #         ct = sweep(initial_vortex,targs,dx,dy,ops,bs,kernel,cpu_source,affinity=aff,filename=fname)
    #         if rank == master_rank:
    #             f.write("Swept: "+str((ct,bs,aff))+"\n")
    #         comm.Barrier()

    # for i,bs in enumerate(block_sizes[:1]):
    #     for j,aff in enumerate(affinities):
    #         fname = decomp_name+"_"+str(i)+"_"+str(j)
    #         ct = decomp(initial_vortex,targs,dx,dy,ops,bs,kernel,cpu_source,affinity=aff,filename=fname)
    #         if rank == master_rank:
    #             f.write("Decom: "+str((ct,bs,aff))+"\n")
    #         comm.Barrier()
    #For testing individual sweep

    ct = sweep(initial_vortex,targs,dx,dy,ops,(10,10,1),kernel,cpu_source,affinity=0.5,filename="./results/temp")



if __name__ == "__main__":
    args = tuple()
    # sm = "Hi,\nYour function run is complete.\n"
    # notifier = NotiPy(test,args,sm,"asw42695@gmail.com",rank=rank,timeout=None)
    # notifier.run()
    test(args)

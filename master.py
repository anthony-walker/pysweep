#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
#Create arguments here for the solver and call them with pst.py
import os
import numpy as np
# from notipy.notipy import NotiPy

class controller(object):
    def __init__(self):
        """Use this to create class that will execute all the appropriate cases."""
        self.np = 16
        self.ssm = "mpiexec -n "+str(self.np)+" python ~/pysweep/src/pst.py "
        self.swept = " swept "
        self.decomp = " standard "
        self.asize = int(4*5*6)
        self.affs = np.arange(0.5,1,0.1)
        self.ts = " -dt 0.01 -tf 5 "
        self.runs = list()
        self.block_sizes()
        self.sizes()
        self.mpi_arg_parse(self.swept)
        self.mpi_arg_parse(self.decomp)

    def __call__(self,args):
        """Use this to execute the cases"""
        for run in self.runs:
            os.system(run)

    def block_sizes(self):
        """Use this function to create blocksizes"""
        self.blocks = list()
        for i in range(8,33,4):
            if self.asize%self.asize==0:
                self.blocks.append(i)

    def sizes(self):
        """Use this function to create a list of array sizes"""
        self.sizes = [self.asize]
        for i in range(5,21,5):
            self.sizes.append(i*self.asize)

    def mpi_arg_parse(self,mode):
        """Use this function to create runs list"""
        for i, affinity in enumerate(self.affs):
            for j, block in enumerate(self.blocks):
                for k, size in enumerate(self.sizes):
                    sstr = self.ssm + mode
                    sstr += " -nx "+str(size)+" -ny "+str(size)
                    sstr += " -X "+str(int(size/10))+" -Y "+str(int(size/10))
                    sstr += " -b " +str(block)+" -a "+str(affinity)
                    sstr += self.ts+" --hdf5 ./results/"+mode+str(i)+str(j)+str(k)+" "
                    self.runs.append(sstr)

exe_cont = controller()
print(exe_cont.runs[0])
print(exe_cont.runs[-1])
print(exe_cont.blocks)
print(exe_cont.sizes)
# sm = "Hi,\nYour function run is complete.\n"
# notifier = NotiPy(exe_cont,tuple(),sm,"asw42695@gmail.com",timeout=None)
# notifier.run()

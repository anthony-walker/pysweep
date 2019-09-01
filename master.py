#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
#Create arguments here for the solver and call them with pst.py
import os
from notipy.notipy import NotiPy

class controller(object):
    def __init__(self,args):
        """Use this to create class that will execute all the appropriate cases."""
        n = args
        self.ssm = "mpiexec -n "+str(n)+" python ~/pysweep/src/pst.py"
        self.swept = " swept"
        self.decomp = "standard"

    def __call__(self,args):
        """Use this to execute the cases"""

        self.cstr = self.ssm+self.swept
        os.system(self.cstr)

    def mpi_arg_parse():
        pass

exe_cont = controller(4)
sm = "Hi,\nYour function run is complete.\n"
notifier = NotiPy(exe_cont,tuple(),sm,"asw42695@gmail.com",timeout=None)
notifier.run()

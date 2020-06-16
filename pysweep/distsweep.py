
#Programmer: Anthony Walker
#PySweep is a package used to implement the swept rule for solving PDEs
import sys, h5py, time, os, shutil, mpi4py.MPI as MPI
pysweep_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pysweep_path)
import gputools.GPUtil as GPUtil

# def dsweep(arr0,gargs,swargs,filename ="results",exid=[],dType='float64'):
#     """Use this function to perform swept rule
#     args:
#     arr0 -  3D numpy array of initial conditions (v (variables), x,y)
#     gargs:   (global args)
#         The first three variables must be time arguments (t0,tf,dt).
#         Subsequent variables can be anything passed to the source code in the
#         "set_globals" function
#     swargs: (Swept args)
#         TSO - order of the time scheme
#         OPS - operating points to remove from each side for a function step
#             (e.g. a 5 point stencil would result in OPS=2).
#         BS - an integer which represents the x and y dimensions of the gpu block size
#         AF -  the GPU affinity (GPU work/CPU work)/TotalWork
#         GS - GPU source code file
#         CS - CPU source code file
#     filename: Name of the output file excluding hdf5
#     exid: GPU ids to exclude from the calculation.
#     dType: a string data type which will be entered into numpy to obtain a datatype object
#     """
#     AF = swargs[3]
#     swnames = ['TSO','OPS','BS','AF']
#     #Getting GPU info
#     #-------------MPI Set up----------------------------#
#     comm = MPI.COMM_WORLD
#     processor = MPI.Get_processor_name()
#     proc_mult = 1
#     rank = comm.Get_rank()  #current rank
#     master_rank = 0
#     # Making sure arr0 has correct data type
#     arr0 = arr0.astype(dType)
#     #Create input file
#     ifn = "input_file.hdf5"
#     hdf5_file = h5py.File(ifn, 'w', driver='mpio', comm=comm)
#     hdf5_file.create_dataset("arr0",arr0.shape,data=arr0)
#     hdf5_file.create_dataset("swargs",(len(swargs[:-2]),),data=swargs[:-2])
#     hdf5_file.create_dataset("gargs",(len(gargs),),data=gargs[:])
#     FN = [ord(ch) for ch in filename]
#     DT = [ord(ch) for ch in dType]
#     hdf5_file.create_dataset("exid",(len(exid),),data=exid)
#     hdf5_file.create_dataset("filename",(len(FN),),data=FN)
#     hdf5_file.create_dataset("dType",(len(DT),),data=DT)
#     comm.Barrier()
#     hdf5_file.close()
#     #Copy GPU/CPU files
#     cpath = os.path.join(os.path.dirname(__file__),'dcore')
#     shutil.copyfile(swargs[5],os.path.join(cpath,'cpu.py'))
#     shutil.copyfile(swargs[4],os.path.join(cpath,'gpu.h'))
#     #Getting GPUs if affinity is greater than 0
#     if AF>0:
#         gpu_rank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=exid,limit=1e8) #getting devices by load
#         num_gpus = len(gpu_rank)
#     else:
#         gpu_rank = []
#         num_gpus = 0
#     node_gpus_list = comm.allgather(num_gpus)
#     if rank == master_rank:
#         #Multipler for Number of processes to add
#         proc_mult += max(node_gpus_list)
#     proc_mult = comm.bcast(proc_mult,root=master_rank)
#     comm.Barrier()
#     num_procs = int(proc_mult*comm.Get_size())
#     # Spawning MPI processes
#     if rank == master_rank:
#         path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dse.py')
#         MPI.COMM_SELF.Spawn(sys.executable,args=[path],maxprocs=num_procs)
#     else:
#         MPI.Finalize()
#         exit(0)
#
#
# if __name__ == "__main__":
#     #Analytical properties
#     import numpy
#     t0,tf,dt,npx,X,nps,aff,blks=(0,0.1,0.001,12,1.2,1,0,12)
#     arr0 = numpy.zeros((4,npx,npx))
#     ct = 0
#     for idz in range(arr0.shape[0]):
#         for idx in range(arr0.shape[1]):
#             for idy in range(arr0.shape[2]):
#                 arr0[idz,idx,idy] = ct
#                 ct += 1
#     #Dimensions and steps
#     dx = X/(npx-1)
#     dy = X/(npx-1)
#     #Changing arguments
#     gargs = (t0,tf,dt,dx,dy,1.4)
#     direc = os.path.abspath(__file__)[:-len('distributed/sweep/distsweep.py')]
#     direc = os.path.join(direc,'equations')
#     swargs = (2,2,blks,0,os.path.join(direc,'eqt.h'),os.path.join(direc,'eqt.py'))
#     dsweep(arr0,gargs,swargs,filename='distsweeptest')

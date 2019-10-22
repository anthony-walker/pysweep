
#Programmer: Anthony Walker
#This file is for global variable initialization
from block import create_dist_up_sets,create_dist_bridge_sets,create_dist_down_sets
from decomposition import create_shared_pool_array
from source import *
import h5py
import os


def init_globals(ops,tso,af,BS,MPSS,grb,GS,CS):
    """Use this function to initialize globals."""
    #Setting global variables
    global OPS,TSO,AF,gts,up_sets, down_sets, oct_sets, x_sets, y_sets
    #Setting common globals
    gts = 0  #Counter for writing on the appropriate step
    OPS = ops
    TSO = tso
    AF = af
    #Setting uncommon globals
    if grb:
        up_sets,down_sets,oct_sets,x_sets,y_sets = None,None,None,None,None
    else:
        up_sets = create_dist_up_sets(BS,OPS)
        down_sets = create_dist_down_sets(BS,OPS)
        oct_sets = down_sets+up_sets
        x_sets,y_sets = create_dist_bridge_sets(BS,OPS,MPSS)


def make_hdf5(filename,cluster_master,comm,rank,BS,arr0,time_steps,AF,dType):
    """Use this function to make hdf5 output file."""
    filename+=".hdf5"
    hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=comm)
    hdf_bs = hdf5_file.create_dataset("BS",(len(BS),),data=BS)
    hdf_as = hdf5_file.create_dataset("array_size",(len(arr0.shape)+1,),data=(time_steps,)+arr0.shape)
    hdf_aff = hdf5_file.create_dataset("AF",(1,),data=AF)
    hdf_time = hdf5_file.create_dataset("time",(1,))
    hdf5_data_set = hdf5_file.create_dataset("data",(time_steps+1,arr0.shape[0],arr0.shape[1],arr0.shape[2]),dtype=dType)
    if rank == cluster_master:
        hdf5_data_set[0,:,:,:] = arr0[:,:,:]
    return hdf5_file

def cpu_core(blocks,shared_shape,OPS,BS,CS,GRB,gargs,MPSS,total_cpu_block,node_comm):
    """Use this function to execute core cpu only processes"""
    blocks = create_es_blocks(blocks,shared_shape,OPS,BS)
    SM = build_cpu_source(CS) #Building Python source code
    SM.set_globals(GRB,SM,*gargs)
    #Creating sets for cpu calculation
    up_sets = create_dist_up_sets(BS,OPS)
    down_sets = create_dist_down_sets(BS,OPS)
    oct_sets = down_sets+up_sets
    x_sets,y_sets = create_dist_bridge_sets(BS,OPS,MPSS)
    return blocks,upsets,down_sets


#Programmer: Anthony Walker
#This file is for global variable initialization
from dcore import block, decomp, sgs
from ccore import source
import h5py, os, GPUtil
import numpy as np
from mpi4py import MPI


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

def get_gpu_info(rank,cluster_master,nid,cluster_comm,AF,BS,exid,processors,ns,arr_shape):
    """Use this function to split data amongst nodes."""
    #Assert that the total number of blocks is an integer
    tnc = cluster_comm.Get_size()
    if AF>0:
        lim = ns if AF==1 else ns-1
        gpu_rank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=exid,limit=lim) #getting devices by load
        num_gpus = len(gpu_rank)
    else:
        gpu_rank = []
        num_gpus = 0
    j = np.arange(0,tnc+1,1,dtype=np.intc) if AF < 1 else np.zeros(tnc,dtype=np.intc)
    ranks,nids,ngl = zip(*[(None,0,0)]+cluster_comm.allgather((rank,nid,num_gpus)))
    ranks = list(ranks)
    ranks[0] = ranks[-1] #Set first rank to last rank
    ranks.append(ranks[1]) #Add first rank to end
    i = [0]+[ngl[i]+sum(ngl[:i]) for i in range(1,len(ngl))]
    tng = np.sum(ngl)
    NB = np.prod(arr_shape[1:])/np.prod(BS)
    NR = arr_shape[1]/BS[0]
    assert (NB).is_integer(), "Provided array dimensions is not divisible by the specified block size."
    GNR = np.ceil(NR*AF)
    CNR = NR-GNR
    k = (GNR-GNR%tng)/tng
    n = GNR%tng
    m = tng-n
    assert (k+1)*n+k*m == GNR, "GPU: Problem with decomposition."
    kc = (CNR-CNR%tnc)/tnc
    nc = CNR%tnc
    mc = tnc-nc
    assert (kc+1)*nc+kc*mc == CNR, "CPU: Problem with decomposition."
    GRF = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x
    CRF = lambda y: (kc)*mc+(kc+1)*(y-mc) if y > mc else (kc)*y
    # print(GRF(i[nid-1]),CRF(j[nid]))
    gL = GRF(i[nid-1])
    gU = GRF(i[nid])
    cL = CRF(j[nid-1])
    cU = CRF(j[nid])
    rLow = gL+cL
    rHigh= gU+cU
    gMag = gU-gL
    cMag = cU-cL
    return gpu_rank,tng,num_gpus,(rLow,rHigh,gMag,cMag),(ranks[nid-1],ranks[nid+1]),GNR,CNR


def find_remove_ranks(node_ranks,AF,num_gpus):
    """Use this function to find ranks that need removed."""
    ranks_to_remove = list()
    # node_ranks = [node_ranks[0]] if AF == 0 else node_ranks
    while len(node_ranks) > num_gpus+(1-int(AF)):
        ranks_to_remove.append(node_ranks.pop())
    return ranks_to_remove


def cpu_core(sarr,total_cpu_block,shared_shape,OPS,BS,CS,GRB,gargs,MPSS):
    """Use this function to execute core cpu only processes"""
    blocks = decomp.create_escpu_blocks(decomp.create_cpu_blocks(total_cpu_block,BS),shared_shape,BS)
    sgs.SM = source.build_cpu_source(CS) #Building Python source code
    sgs.SM.set_globals(GRB,sgs.SM,*gargs)
    #Creating sets for cpu calculation
    sgs.up_sets = block.create_dist_up_sets(BS,OPS)
    sgs.down_sets = block.create_dist_down_sets(BS,OPS)
    sgs.oct_sets = sgs.down_sets+sgs.up_sets
    sgs.y_sets,sgs.x_sets = block.create_dist_bridge_sets(BS,OPS,MPSS)
    print(sarr[total_cpu_block].shape)
    sgs.carr = decomp.create_shared_pool_array(sarr[total_cpu_block].shape)
    sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    return blocks,total_cpu_block

def gpu_core(blocks,BS,OPS,GS,CS,gargs,GRB,MPSS,MOSS,TSO):
    """Use this function to execute core gpu only processes"""
    block_shape = [i.stop-i.start for i in blocks]
    block_shape[-1] += int(2*BS[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    GRD = (int((block_shape[2])/BS[0]),int((block_shape[3])/BS[1]))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SGIDS = (BS[0]+2*OPS)*(BS[1]+2*OPS)
    STS = SGIDS*NV #Shared time shift
    VARS =  block_shape[2]*(block_shape[3])
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})
    garr = decomp.create_local_gpu_array(block_shape)
    # print(block_shape)
    #Building CUDA source code
    sgs.SM = source.build_gpu_source(GS,os.path.basename(__file__))
    source.swept_constant_copy(sgs.SM,const_dict)
    cpu_SM = source.build_cpu_source(CS)   #Building cpu source for set_globals
    cpu_SM.set_globals(GRB,sgs.SM,*gargs)
    del cpu_SM
    return block_shape,GRD,garr

def mpi_destruction(rank,node_ranks,comm,ranks_to_remove,all_ranks):
    """Use this to destory unwanted mpi processes."""
    [all_ranks.remove(x) for x in set([x[0] for x in comm.allgather(ranks_to_remove) if x])]
    comm.Barrier()
    #Remaking comms
    if rank in all_ranks:
        #New Global comm
        ncg = comm.group.Incl(all_ranks)
        comm = comm.Create_group(ncg)
        # New node comm
        node_group = comm.group.Incl(node_ranks)
        node_comm = comm.Create_group(node_group)
    else: #Ending unnecs
        MPI.Finalize()
        exit(0)
    comm.Barrier()
    return node_comm,comm

def block_dissem(rank,node_master,shared_shape,rows_per_gpu,BS,num_gpus,OPS,node_ranks,gpu_rank,node_comm):
    """Use this function to spread blocks to ranks."""
    #Creating blocks to be solved
    if rank==node_master:
        gpu_blocks,cpu_blocks,total_cpu_block = decomp.create_blocks(shared_shape,rows_per_gpu,BS,num_gpus,OPS)
        gpu_ranks = node_ranks[:num_gpus]
        cpu_ranks = node_ranks[num_gpus:]
        blocks = np.array_split(gpu_blocks,num_gpus) if gpu_blocks else gpu_blocks
        blocks.append(cpu_blocks) if cpu_blocks else None
        gpu_rank += [(False,None) for i in range(len(cpu_ranks))]
        node_data = zip(blocks,gpu_rank)
    else:
        node_data = None
        total_cpu_block = None
    blocks,gpu_rank =  node_comm.scatter(node_data)
    total_cpu_block = node_comm.bcast(total_cpu_block)
    GRB = gpu_rank[0]
    return GRB,blocks,total_cpu_block,gpu_rank

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
    j = np.arange(0,tnc+1,1,dtype=np.intc) if AF < 1 else np.zeros(tnc+1,dtype=np.intc)
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
    k = (GNR-GNR%tng)/tng if tng!=0 else 0.0
    n = GNR%tng if tng!=0 else 0.0
    m = tng-n
    assert (k+1)*n+k*m == GNR, "GPU: Problem with decomposition."
    kc = (CNR-CNR%tnc)/tnc
    nc = CNR%tnc
    mc = tnc-nc
    assert (kc+1)*nc+kc*mc == CNR, "CPU: Problem with decomposition."
    GRF = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x
    CRF = lambda y: (kc)*mc+(kc+1)*(y-mc) if y > mc else (kc)*y
    gL = GRF(i[nid-1])
    gU = GRF(i[nid])
    cL = CRF(j[nid-1])
    cU = CRF(j[nid])
    rLow = gL+cL
    rHigh= gU+cU
    gMag = gU-gL
    cMag = cU-cL
    return gpu_rank,tng,num_gpus,(rLow,rHigh,gMag,cMag),(ranks[nid-1],ranks[nid+1]),GNR,CNR

def find_remove_ranks(node_ranks,AF,num_gpus,CNR):
    """Use this function to find ranks that need removed."""
    ranks_to_remove = list()
    # node_ranks = [node_ranks[0]] if AF == 0 else node_ranks
    try:
        assert CNR > 0 if AF < 1 else True, "AssertionError: CNR!>0, Affinity specified forces number of rows for the cpu to be zero, CPU process will be destroyed. Adjusting affintiy to 1."
        while len(node_ranks) > num_gpus+(1-int(AF)):
            ranks_to_remove.append(node_ranks.pop())
    except Exception as e:
        print(str(e))
        while len(node_ranks) > num_gpus:
            ranks_to_remove.append(node_ranks.pop())
    return ranks_to_remove

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

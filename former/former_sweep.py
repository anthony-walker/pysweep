def archs_phase_1(block_size,num_cpu,num_gpu):
    """Use this function to determine the array splits for the first phase (grid1)"""
    #Axes
    x_ax = 1
    y_ax = 2
    #Getting shape of grid
    grid_shape = np.shape(grid)
    #Creating work element partition integers
    par_x = int(grid_shape[x_ax]/block_size[x_ax])
    par_y = int(grid_shape[y_ax]/block_size[y_ax])
    #Split in y
        #Split in x
            #Add to list

def archs_phase_2(block_size,num_cpu,num_gpu):
    """Use this function to determine the array splits for the second phase (grid2)"""
    #Axes
    x_ax = 1
    y_ax = 2
    #Getting shape of grid
    grid_shape = np.shape(grid)
    #Creating work element partition integers
    par_x = int(grid_shape[x_ax]/block_size[x_ax])
    par_y = int(grid_shape[y_ax]/block_size[y_ax])
    # print("Par(x,y,t): ",par_x,", ",par_y,", ",par_t) #Printing partitions

### ----------------------------- COMPLETED FUNCTIONS -----------------------###
def arch_work_blocks(plane_shape,block_size,gpu_affinity):
    """Use to determine the number of blocks for gpus vs cpus."""
    blocks_x = int(plane_shape[0]/block_size[0])
    blocks_y = int(plane_shape[1]/block_size[1])
    total_blocks = blocks_x*blocks_y
    gpu_blocks = round(total_blocks/(1+1/gpu_affinity))
    block_mod_y = gpu_blocks%blocks_y
    if  block_mod_y!= 0:
        gpu_blocks+=blocks_y-block_mod_y
    cpu_blocks = total_blocks-gpu_blocks
    return (gpu_blocks,cpu_blocks)

def arch_query():
    """Use this method to query the system for information about its hardware"""
    #-----------------------------MPI setup--------------------------------
    comm = MPI.COMM_WORLD
    master_rank = 0 #master rank
    num_ranks = comm.Get_size() #number of ranks
    rank = comm.Get_rank()  #current rank
    rank_info = None    #Set to none for later gather
    cores = mp.cpu_count()  #getting cores of each rank with multiprocessing package

                            #REMOVE ME AFTER TESTING
    #####################################################################
    #Subtracting cores for those used by virtual cluster.
    # if rank == master_rank:
    #     cores -= 14
    #####################################################################

    #Getting avaliable GPUs with GPUtil
    gpu_ids = GPUtil.getAvailable(order = 'load',excludeID=[1],limit=10)
    gpus = 0
    gpu_id = None
    if gpu_ids:
        gpus = len(gpu_ids)
        gpu_id = gpu_ids[0]
    #Gathering all of the information to the master rank
    rank_info = comm.gather((rank, cores, gpus,gpu_id),root=0)
    if rank == master_rank:
        gpu_sum = 0
        cpu_sum = 0
        #Getting total number of cpu's and gpu
        for ri in rank_info:
            cpu_sum += ri[1]
            gpu_sum += ri[2]
            if ri[-1] is not None:
                gpu_id = ri[3]
        return gpu_sum, cpu_sum, gpu_id
    return None,None,gpu_id


    # #----------------------------Creating shared arrays-------------------------#
    # global cpu_array
    # cpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    # cpu_array = np.ctypeslib.as_array(cpu_array_base.get_obj())
    # cpu_array = cpu_array.reshape(block_size)
    #
    # global gpu_array
    # gpu_array_base = mp.Array(ctypes.c_double, shm_dim)
    # gpu_array = np.ctypeslib.as_array(gpu_array_base.get_obj())
    # gpu_array = gpu_array.reshape(block_size)


def CPU_UpPyramid(args):
    """Use this function to solve a block on the CPU."""
    block,source_mod,ops = args
    plane_shape = block.shape
    bsx = block.shape[2]
    bsy = block.shape[3]
    iidx = list(np.ndindex(plane_shape[2:]))
    #Bounds
    lb = 0
    ub = [plane_shape[2],plane_shape[3]]
    #Going through all swept steps
    # pts = [iidx]    #This is strictly for debugging
    while ub[0] >= lb and ub[1] >= lb:

        lb += ops
        ub = [x-ops for x in ub]
        iidx = [x for x in iidx if x[0]>=lb and x[1]>=lb and x[0]<ub[0] and x[1]<ub[1]]
        print(lb,ub)
        print(iidx)
        if iidx:
            block = source_mod.step(block,iidx,0)
            # pts.append(iidx) #This is strictly for debuggings
    # return pts #This is strictly for
    return block


def rank_split(arr0,rank_size):
    """Use this function to equally split data among the ranks"""
    major_axis = plane_shape.index(max(plane_shape))
    return np.array_split(arr0,rank_size,axis=major_axis)

#--------------------------------NOT USED CURRENTLY-----------------------------
def edges(arr,ops,shape_adj=-1):
    """Use this function to generate boolean arrays for edge handling."""
    mask = np.zeros(arr.shape[:shape_adj], dtype=bool)
    mask[(arr.ndim+shape_adj)*(slice(ops, -ops),)] = True
    return mask

def dummy_fcn(arr):
    """This is a testing function for arch_speed_comp."""
    iidx = np.ndindex(np.shape(arr))
    for i,idx in enumerate(iidx):
        arr[idx] *= i
        # print(i,value)
    return arr
def create_map():
    """Use this function to create local maps for process communication."""
    smap = set() #Each rank has its own map

def extended_shape(orig_shape,block_size):
    """Use this function to develop extended array shapes."""
    return (orig_shape[0],orig_shape[1]+int(block_size[0]/2),orig_shape[2])
def CPU_UpPyramid(args):
    """Use this function to solve a block on the CPU."""
    block,source_mod,ops = args
    b_shape_x = block.shape[2]
    b_shape_y = block.shape[3]
    iidx = tuple(np.ndindex(block.shape[2:]))
    #Removing elements for swept step
    ts = 0
    while len(iidx)>0:
        #Adjusting indices
        tl = tuple()
        iidx = iidx[ops*b_shape_x:-ops*b_shape_x]
        b_shape_y-=2*ops
        for i in range(1,b_shape_y+1,1):
            tl+=iidx[i*b_shape_x-b_shape_x+ops:i*b_shape_x-ops]
        b_shape_x-=2*ops
        iidx = tl
        #Calculating Step
        block = source_mod.step(block,iidx,ts)
        ts+=1
    return block

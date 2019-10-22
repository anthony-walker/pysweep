#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.

def FirstPrism(sarr,garr,blocks,total_cpu_block,up_sets,x_sets,gts,mpi_pool,pargs):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        arr_gpu = cuda.mem_alloc(garr.nbytes)
        garr = copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        mpi_pool.map(dCPU_UpPyramid,zip(cblocks,repeat(gts)))
        mpi_pool.map(dCPU_Ybridge,zip(xblocks,repeat(gts)))
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = carr[:,:,:,:]

def UpPrism(sarr,garr,blocks,total_cpu_block,up_sets,x_sets,gts,mpi_pool,pargs):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    SM,GRB,BS,GRD,OPS,TSO,ssb = pargs
    #Splitting between cpu and gpu
    if GRB:
        arr_gpu = cuda.mem_alloc(garr.nbytes)
        garr = copy_s_to_g(sarr,garr,blocks,BS)
        cuda.memcpy_htod(arr_gpu,garr)
        SM.get_function("UpPyramid")(arr_gpu,np.int32(gts),grid=GRD, block=BS,shared=ssb)
        SM.get_function("YBridge")(arr_gpu,np.int32(gts),grid=(GRD[0],GRD[1]-1), block=BS,shared=ssb)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,arr_gpu)
        pm(garr,2,'%.0f')
        sarr[blocks]=garr[:,:,:,BS[0]:-BS[0]]
    else:   #CPUs do this
        cblocks,xblocks = zip(*blocks)
        mpi_pool.map(dCPU_UpPyramid,zip(cblocks,repeat(gts)))
        mpi_pool.map(dCPU_Ybridge,zip(xblocks,repeat(gts)))
        #Copy result to MPI shared process array
        sarr[total_cpu_block] = carr[:,:,:,:]


def dCPU_UpPyramid(block):
    """Use this function to build the Up Pyramid."""
    #UpPyramid of Swept Step
    global carr
    block,ct = block
    for ts,swept_set in enumerate(up_sets,start=TSO-1):
        #Calculating Step
        carr[block] = SM.step(carr[block],swept_set,ts,ct)
        ct+=1

def dCPU_Ybridge(block):
    """Use this function to build the XBridge."""
    global carr,TSO
    block,ct = block
    for ts,swept_set in enumerate(x_sets,start=TSO-1):
        #Calculating Step
        carr[block] = SM.step(carr[block],swept_set,ts,ct)
        ct+=1

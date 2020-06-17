
import pysweep.core.functions as functions

def swept_cpu_core(sarr,total_cpu_block,shared_shape,OPS,BS,gargs,MPSS,TSO):
    """Use this function to execute core cpu only processes"""
    blocks,total_cpu_block = decomp.create_cpu_blocks(total_cpu_block,BS,shared_shape)
    blocks = decomp.create_escpu_blocks(blocks,shared_shape,BS)
    cpu.set_globals(False,*gargs)
    #Creating sets for cpu calculation
    up_sets = block.create_dist_up_sets(BS,OPS)
    down_sets = block.create_dist_down_sets(BS,OPS)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = block.create_dist_bridge_sets(BS,OPS,MPSS)
    #Shared array
    sgs.carr = decomp.create_shared_pool_array(sarr[total_cpu_block].shape)
    sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    #Create function objects
    Up = functions.GeometryCPU(0,up_sets,TSO,MPSS,TSO-1)
    Down = functions.GeometryCPU(0,down_sets,TSO,MPSS,TSO-1)
    Xb = functions.GeometryCPU(0,x_sets,TSO,MPSS,TSO-1)
    Yb = functions.GeometryCPU(0,y_sets,TSO,MPSS,TSO-1)
    Oct = functions.GeometryCPU(0,oct_sets,TSO,MPSS,TSO-1)
    return blocks,total_cpu_block,Up,Down,Oct,Xb,Yb

def swept_gpu_core(blocks,BS,OPS,gargs,GRB,MPSS,MOSS,TSO):
    """Use this function to execute core gpu only processes"""
    block_shape = [i.stop-i.start for i in blocks]
    block_shape[-1] += int(2*BS[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    GRD = (int((block_shape[2])/BS[0]),int((block_shape[3])/BS[1]))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SX = block_shape[2]
    SY = block_shape[3]
    VARS =  int(SX*SY)
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,'SX':SX,'SY':SY})
    garr = decomp.create_local_gpu_array(block_shape)
    garr = cuda.mem_alloc(garr.nbytes)
    #Building CUDA source code
    SM = source.build_gpu_source(os.path.dirname(__file__))
    source.swept_constant_copy(SM,const_dict)
    cpu.set_globals(GRB,*gargs,source_mod=SM)
    # Make GPU geometry
    Up = functions.GeometryGPU(0,SM.get_function("UpPyramid"),BS,GRD,MPSS,block_shape)
    Down = functions.GeometryGPU(0,SM.get_function("DownPyramid"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    Yb = functions.GeometryGPU(0,SM.get_function("YBridge"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    Xb = functions.GeometryGPU(0,SM.get_function("XBridge"),BS,GRD,MPSS,block_shape)
    Oct = functions.GeometryGPU(0,SM.get_function("Octahedron"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    return SM,garr,Up,Down,Oct,Xb,Yb

def standard_cpu_core(sarr,total_cpu_block,shared_shape,OPS,BS,GRB,TSO,gargs):
    """Use this function to execute core cpu only processes"""
    xslice = slice(shared_shape[2]-OPS-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2]-OPS,1)
    swb = (total_cpu_block[0],total_cpu_block[1],xslice,total_cpu_block[3])
    blocks,total_cpu_block = decomp.create_cpu_blocks(total_cpu_block,BS,shared_shape,OPS)
    cpu.set_globals(GRB,*gargs,source_mod=None)
    #Creating sets for cpu calculation
    DecompObj = functions.DecompCPU(0,[(x+OPS,y+OPS) for x,y in numpy.ndindex((BS[0],BS[1]))],TSO-1,cwrite=swb,cread=total_cpu_block)
    sgs.carr = decomp.create_shared_pool_array(sarr[total_cpu_block].shape)
    return DecompObj,blocks

def standard_gpu_core(blocks,BS,OPS,gargs,GRB,TSO):
    """Use this function to execute core gpu only processes"""
    block_shape = [i.stop-i.start for i in blocks[:2]]+[i.stop-i.start+2*OPS for i in blocks[2:]]
    gwrite = blocks
    gread = blocks[:2]
    gread+=tuple([slice(i.start-OPS,i.stop+OPS,1) for i in blocks[2:]])
    # Creating local GPU array with split
    GRD = (int((block_shape[2]-2*OPS)/BS[0]),int((block_shape[3]-2*OPS)/BS[1]))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SX = block_shape[2]
    SY = block_shape[3]
    VARS =  block_shape[2]*(block_shape[3])
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"VARS":VARS,"TIMES":TIMES,"OPS":OPS,"TSO":TSO,"SX":SX,"SY":SY})
    garr = decomp.create_local_gpu_array(block_shape)
    #Building CUDA source code
    SM = source.build_gpu_source(os.path.dirname(__file__))
    source.decomp_constant_copy(SM,const_dict)
    cpu.set_globals(GRB,*gargs,source_mod=SM)
    DecompObj  = functions.DecompGPU(0,SM.get_function("Decomp"),GRD,BS,gread,gwrite,garr)
    return DecompObj,garr

#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing process management
# and data decomposition for the swept rule.
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#MPI imports
from mpi4py import MPI
import importlib
#System imports
import os.path as op
import inspect

def create_CPU_sarray(comm,arr_shape,dType,arr_bytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(dType.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arr_bytes, itemsize, comm=comm)
    shared_buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=shared_buf, dtype=dType.type, shape=arr_shape)
    return arr

def get_affinity_slices(affinity,block_size,arr_shape):
    """Use this function to split the given data based on rank information and the affinity.
    affinity -  a value between zero and one. (GPU work/CPU work)/Total Work
    block_size -  gpu block size
    arr_shape - shape of array initial conditions array (v,x,y)
    ***This function splits to the nearest column, so the affinity may change***
    """
    #Getting number of GPU blocks based on affinity
    blocks_per_column = arr_shape[2]/block_size[1]
    blocks_per_row = arr_shape[1]/block_size[0]
    num_blocks = int(blocks_per_row*blocks_per_column)
    gpu_blocks = round(affinity*num_blocks)
    #Rounding blocks to the nearest column
    col_mod = gpu_blocks%blocks_per_column  #Number of blocks ending in a column
    col_perc = col_mod/blocks_per_column    #What percentage of the column
    gpu_blocks += round(col_perc)*blocks_per_column-col_mod #Round to the closest column and add the appropriate value
    #Getting number of columns and rows
    num_columns = int(gpu_blocks/blocks_per_column)
    #Region Indicies
    gpu_slices = (slice(0,arr_shape[0],1),slice(0,int(block_size[0]*num_columns),1),)
    cpu_slices = (slice(0,arr_shape[0],1),slice(int(block_size[0]*num_columns),arr_shape[1],1),slice(0,arr_shape[2],1))
    return gpu_slices, cpu_slices

def boundary_update(shared_arr,ops,SPLITX,SPLITY):
    """Use this function to update the boundary point ghost nodes."""
    SX = SPLITX+ops
    SY = SPLITY+ops
    #Back to front
    shared_arr[:,:,:ops,:] = shared_arr[:,:,-SX-ops:-SX,:]
    shared_arr[:,:,:,:ops] = shared_arr[:,:,:,-SY-ops:-SY]
    #Front to back
    shared_arr[:,:,-SX:,:] = shared_arr[:,:,ops:SX+ops,:]
    shared_arr[:,:,:,-SY:] = shared_arr[:,:,:,ops:SY+ops]

def create_write_region(comm,rank,master,total_ranks,block_size,arr_shape,slices,time_steps,ops):
    """Use this function to split regions amongst the architecture."""
    y_blocks = arr_shape[2]/block_size[1]
    blocks_per_rank = round(y_blocks/total_ranks)
    rank_blocks = (blocks_per_rank*(rank),blocks_per_rank*(rank+1))
    rank_blocks = comm.gather(rank_blocks,root=master)
    if rank == master:
        rem = int(y_blocks-rank_blocks[-1][1])
        if rem > 0:
            ct = 0
            for i in range(rem):
                rank_blocks[i] = (rank_blocks[i][0]+ct,rank_blocks[i][1]+1+ct)
                ct +=1

            for j in range(rem,total_ranks):
                rank_blocks[j] = (rank_blocks[j][0]+ct,rank_blocks[j][1]+ct)
        elif rem < 0:
            ct = 0
            for i in range(rem,0,1):
                rank_blocks[i] = (rank_blocks[i][0]+ct,rank_blocks[i][1]-1+ct)
                ct -=1
    rank_blocks = comm.scatter(rank_blocks,root=master)
    x_slice = slice(slices[1].start+ops,slices[1].stop+ops,1)
    y_slice = slice(int(block_size[1]*rank_blocks[0]+ops),int(block_size[1]*rank_blocks[1]+ops),1)
    return (slice(0,time_steps,1),slices[0],x_slice,y_slice)

def create_boundary_regions(wr,SPLITX,SPLITY,ops,ss,bridge_slices):
    """Use this function to create boundary write regions."""
    boundary_regions = tuple()
    eregions = tuple()
    x_regions = tuple()
    y_regions = tuple()
    region_start = wr[:2]
    c1 = wr[2].start==ops
    c2 = wr[3].start==ops
    sx = ss[2]-ops-SPLITX
    ox = sx-ops
    sy = ss[3]-ops-SPLITY
    oy = sy-ops
    x_reg = slice(ops,wr[2].stop-wr[2].start+ops,1)
    y_reg = slice(ops,wr[3].stop-wr[3].start+ops,1)
    tops = 2*ops
    if c1: #Top edge -  periodic x
        #Boundaries for up pyramid and octahedron
        boundary_regions += (region_start+(slice(ops,SPLITX+tops,1),y_reg,slice(sx,ss[2],1),wr[3]),)
        yer = slice(wr[3].start,ss[3],1) if wr[3].stop+SPLITY+ops == ss[3] else slice(wr[3].start,wr[3].stop,1)
        eregions += (region_start+(slice(ops,SPLITX+ops,1),yer,slice(sx,sx+SPLITX,1),yer,),)
        for x,y in bridge_slices[1]:
            tfxe = x.stop+ox
            xc = tfxe < ss[2]
            tfxe = tfxe if xc else ss[2]
            tfx = slice(x.start+ox,tfxe,1)
            nx = x if xc else slice(x.start,tfx.stop-tfx.start+x.start,1)
            tfys = wr[3].start-ops+y.start+SPLITY
            tfy = slice(tfys,tfys+(y.stop-y.start),1)
            x_regions += ((nx,y,tfx,tfy),)
    if c2: #Side edge -  periodic y
        boundary_regions += (region_start+(x_reg,slice(ops,SPLITY+tops,1),wr[2],slice(sy,ss[3],1)),)
        xer = slice(wr[2].start,ss[2],1) if wr[2].stop+SPLITY+ops == ss[2] else slice(wr[2].start,wr[2].stop,1)
        eregions += (region_start+(xer,slice(ops,SPLITY+ops,1),xer,slice(sy,sy+SPLITY,1)),)
        for x,y in bridge_slices[0]:
            #Finding forward bridge
            tfye = y.stop+ox
            yc = tfye < ss[3]
            tfye = tfye if yc else ss[3]
            tfy = slice(y.start+oy,tfye,1)
            ny = y if yc else slice(y.start,tfy.stop-tfy.start+y.start,1)
            tfxs = wr[2].start-ops+x.start+SPLITX
            tfx = slice(tfxs,tfxs+(x.stop-x.start),1)
            y_regions += ((x,ny,tfx,tfy),)
    if c1 and c2:
        boundary_regions += (region_start+(slice(ops,SPLITX+tops,1),slice(ops,SPLITY+tops,1),slice(sx,ss[2],1),slice(sy,ss[3],1)),)
        eregions += (region_start+(slice(ops,SPLITX+ops,1),slice(ops,SPLITY+ops,1),slice(sx,sx+SPLITX,1),slice(sy,sy+SPLITY,1)),)
        #A bridge can never be on a corner so there is not bridge communication here
    return boundary_regions,eregions,x_regions,y_regions

def create_rev_bridges(wr,SPLITX,SPLITY,ops,ss,bridge_slices):
    """Use this function to create boundary write regions."""
    x_regions = tuple()
    y_regions = tuple()
    sx = ss[2]-ops-SPLITX
    sy = ss[3]-ops-SPLITY
    region_start = wr[:2]
    c1 = wr[2].stop==sx
    c2 = wr[3].stop==sy
    tops = 2*ops
    if c1: #Top edge -  periodic x
        for x,y in bridge_slices[1]:
            xc = x.start > SPLITX
            nx = x if xc else slice(SPLITX,x.stop,1)
            tfxs = 0+nx.start-SPLITX
            tfx = slice(tfxs,tfxs+(nx.stop-nx.start),1)
            tfys = wr[3].start-ops+y.start
            tfy = slice(tfys,tfys+(y.stop-y.start),1)
            x_regions += ((nx,y,tfx,tfy),)
    if c2: #Side edge -  periodic y
        for x,y in bridge_slices[0]:
            yc = y.start > SPLITY
            ny = y if yc else slice(SPLITY,y.stop,1)
            tfys = 0+ny.start-SPLITY
            tfy = slice(tfys,tfys+(ny.stop-ny.start),1)
            tfxs = wr[2].start-ops+x.start
            tfx = slice(tfxs,tfxs+(x.stop-x.start),1)
            y_regions += ((x,ny,tfx,tfy),)
    return x_regions,y_regions



def create_shift_regions(wregion,SPLITX,SPLITY,shared_shape,ops):
    """Use this function to create a shifted region(s)."""
    #Conditions
    sregion = wregion[:2]
    swr = wregion[:2]
    sps = (SPLITX,SPLITY)
    for i, rs in enumerate(wregion[2:]):
        sregion+=(slice(rs.start+sps[i],rs.stop+sps[i],1),)
        swr+=(slice(rs.start+sps[i]+ops,rs.stop+sps[i]-ops,1),)

    xbregion =  sregion[:3]+(wregion[3],)
    ybregion =  sregion[:2]+(wregion[2],sregion[3])
    return sregion,swr,xbregion,ybregion

def create_read_region(region,ops):
    """Use this function to obtain the regions to for reading and writing
        from the shared array. region 1 is standard region 2 is offset by split.
        Note: The rows are divided into regions. So, each rank gets a row or set of rows
        so to speak. This is because affinity split is based on columns.
    """
    #Read Region
    new_region = region[:2]
    new_region += slice(region[2].start-ops,region[2].stop+ops,1),
    new_region += slice(region[3].start-ops,region[3].stop+ops,1),
    return new_region

def get_slices_shape(slices):
    """Use this function to convert slices into a shape tuple."""
    stuple = tuple()
    for s in slices:
        stuple+=(s.stop-s.start,)
    return stuple

def create_local_array(shared_arr,region,dType):
    """Use this function to generate the local arrays from regions."""
    local_shape = get_slices_shape(region)
    local_array = np.zeros(local_shape,dtype=dType)
    local_array[:,:,:,:] = shared_arr[region]
    return local_array


def build_cpu_source(cpu_source):
    """Use this function to build source module from cpu code."""
    module_name = cpu_source.split("/")[-1]
    module_name = module_name.split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, cpu_source)
    source_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_mod)
    return source_mod

def build_gpu_source(kernel_source):
    """Use this function to build the given and swept source module together.
    """
    #GPU Swept Calculations
    #----------------Reading In Source Code-------------------------------#
    file = inspect.getfile(build_cpu_source)
    fname = file.split("/")[-1]
    fpath = op.abspath(inspect.getabsfile(build_cpu_source))[:-len(fname)]+"sweep.h"
    source_code = source_code_read(fpath)
    split_source_code = source_code.split("//!!(@#\n")
    source_code = split_source_code[0]+"\n"+source_code_read(kernel_source)+"\n"+split_source_code[1]
    source_mod = SourceModule(source_code)#,options=["--ptxas-options=-v"])
    return source_mod

def source_code_read(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source

def constant_copy(source_mod,const_dict,add_const=None):
    """Use this function to copy constant args to cuda memory.
        source_mod - the source module obtained from pycuda and source code
        const_dict - dictionary of constants where the key is the global
        add_const - additional dictionary with constants. Note, they should have
                    the correct type.
    """
    #Functions to cast data to appropriate gpu types
    int_cast = lambda x:np.int32(x)
    float_cast = lambda x:np.float32(x)
    casters = {type(0.1):float_cast,type(1):int_cast}
    #Generating constants
    for key in const_dict:
        c_ptr,_ = source_mod.get_global(key)
        cst = const_dict[key]
        cuda.memcpy_htod(c_ptr,casters[type(cst)](cst))

    for key in add_const:
        c_ptr,_ = source_mod.get_global(key)
        cuda.memcpy_htod(c_ptr,add_const[key])

def create_up_sets(block_size,ops):
    """Use this function to create uppyramid sets."""
    bsx = block_size[0]+2*ops
    bsy = block_size[1]+2*ops
    ly = ops
    uy = bsy-ops
    min_bs = int(min(bsx,bsy)/(2*ops))
    iidx = tuple(np.ndindex((bsx,bsy)))
    idx_sets = tuple()
    for i in range(min_bs):
        iidx = iidx[ops*(bsy-i*2*ops):-ops*(bsy-i*2*ops)]
        iidx = [(x,y) for x,y in iidx if y >= ly and y < uy]
        if len(iidx)>0:
            idx_sets+=(iidx,)
        ly+=ops
        uy-=ops
    return idx_sets

def create_down_sets(block_size,ops):
    """Use this function to create the down pyramid sets from up sets."""
    bsx = int(block_size[0]/2)
    bsy = int(block_size[1]/2)
    #Limits
    lx =int((bsx)-ops); #lower x
    ly = int((bsy)-ops); #lower y
    ux = int((bsx)+ops); #upper x
    uy = int((bsy)+ops); #upper y
    #Creating points
    iidx = tuple()
    sets = tuple()
    fset = tuple()
    for i in range(ops,block_size[0]+ops):
        temp = tuple()
        for j in range(ops,block_size[1]+ops):
            temp += (i,j),
            fset += (i,j),
        iidx += temp,
    #Creating sets
    while(ux < block_size[0] and uy < block_size[1]):
        new_set = tuple()
        for row in iidx[lx:ux]:
            for item in row[ly:uy]:
                new_set+=item,
        sets+= new_set,
        lx-=ops
        ly-=ops
        ux+=ops
        uy+=ops
    #Adding final set
    sets += fset,
    return sets

def create_bridge_sets(block_size,ops,MPSS):
    """Use this function to create the iidx sets for bridges."""
    # if mbx == mby:
    bsx = block_size[0]+2*ops
    bsy = block_size[1]+2*ops
    ly = ops+ops   #This first block with be in ops, plus the ghost points
    lx = int(block_size[0]/2)    #This first block with be in ops, plus the ghost points
    uy = ly+block_size[1]-2*ops
    ux = lx+2*ops
    min_bs = int(min(bsx,bsy)/(2*ops))
    iidx = tuple(np.ndindex((bsx,bsy)))
    riidx = [iidx[(x)*bsy:(x+1)*bsy] for x in range(bsx)]
    xbs = tuple()
    ybs = tuple()
    x_bridge = tuple()
    for i in range(MPSS-1):
        temp = tuple()
        xbs += (slice(lx,ux,1),slice(ly,uy,1)),
        ybs += (slice(ly,uy,1),slice(lx,ux,1)),
        for row in (riidx[lx:ux]):
            temp+=row[ly:uy][:]
        x_bridge+=temp,
        lx-=ops
        ux+=ops
        ly+=ops
        uy-=ops
    #Y_bridge
    y_bridge = tuple()
    for bridge in x_bridge:
        temp = tuple()
        for item in bridge:
            temp+=(item[::-1],)
        y_bridge+=(temp,)
    # elif mbx > mby: #This need to be added for non-square blocks then kernels need updated
    #     pass
    # elif mbx < mby:
    #     pass
    return (x_bridge, y_bridge), (xbs, ybs)


def create_blocks_list(arr_shape,block_size,ops):
    """Use this function to create a list of blocks from the array."""
    bsx = int((arr_shape[2]-2*ops)/block_size[0])
    bsy =  int((arr_shape[3]-2*ops)/block_size[1])
    slices = []
    c_slice = (slice(0,arr_shape[0],1),slice(0,arr_shape[1],1),)
    for i in range(ops+block_size[0],arr_shape[2],block_size[0]):
        for j in range(ops+block_size[1],arr_shape[3],block_size[1]):
            t_slice = c_slice+(slice(i-block_size[0]-ops,i+ops,1),slice(j-block_size[1]-ops,j+ops,1))
            slices.append(t_slice)
    return slices


def rebuild_blocks(arr,blocks,local_regions,ops):
    """Use this function to rebuild the blocks."""
    #Rebuild blocks into array
    if len(blocks)>1:
        for ct,lr in enumerate(local_regions):
            lr2 = slice(lr[2].start+ops,lr[2].stop-ops,1)
            lr3 = slice(lr[3].start+ops,lr[3].stop-ops,1)
            arr[:,:,lr2,lr3] = blocks[ct][:,:,ops:-ops,ops:-ops]
        return arr
    else:
        return blocks[0]

def edge_shift(shared_arr,eregions, dir):
    """Use this function to shift edge points>"""
    if dir:
        for bt,bv,x1,y1,x2,y2 in eregions:  #Back to Front
            shared_arr[bt,bv,x1,y1] = shared_arr[bt,bv,x2,y2]
    else:
        for bt,bv,x1,y1,x2,y2 in eregions:  #Front to Back
            shared_arr[bt,bv,x2,y2] = shared_arr[bt,bv,x1,y1]

def hdf_swept_write(shared_arr,reg,hdf_set,hr,MPSS,GST):
    """Use this function to write to the hdf file and shift the shared array
        # data after writing."""
    hdf_set[MPSS*(GST-1)+1:MPSS*(GST)+1,hr[0],hr[1],hr[2]] = shared_arr[0:MPSS,reg[1],reg[2],reg[3]]
    nte = shared_arr.shape[0]-(MPSS)
    shared_arr[:nte,reg[1],reg[2],reg[3]] = shared_arr[MPSS:,reg[1],reg[2],reg[3]]
    shared_arr[nte:,reg[1],reg[2],reg[3]] = 0
    #Do edge comm after this function

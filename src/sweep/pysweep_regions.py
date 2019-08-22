#Programmer: Anthony Walker
#This file contains all the necessary functions to create the regions for the swept rule.

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
    region_start = wr[:2]
    c1 = wr[2].start==ops
    c2 = wr[3].start==ops
    sx = ss[2]-ops-SPLITX
    sy = ss[3]-ops-SPLITY
    x_reg = slice(ops,wr[2].stop-wr[2].start+ops,1)
    y_reg = slice(ops,wr[3].stop-wr[3].start+ops,1)
    tops = 2*ops
    if c1: #Top edge -  periodic x
        #Boundaries for up pyramid and octahedron
        boundary_regions += (region_start+(slice(ops,SPLITX+tops,1),y_reg,slice(sx,ss[2],1),wr[3]),)
        yer = slice(wr[3].start,ss[3],1) if wr[3].stop+SPLITY+ops == ss[3] else slice(wr[3].start,wr[3].stop,1)
        eregions += (region_start+(slice(ops,SPLITX+ops,1),yer,slice(sx,sx+SPLITX,1),yer,),)
    if c2: #Side edge -  periodic y
        boundary_regions += (region_start+(x_reg,slice(ops,SPLITY+tops,1),wr[2],slice(sy,ss[3],1)),)
        xer = slice(wr[2].start,ss[2],1) if wr[2].stop+SPLITY+ops == ss[2] else slice(wr[2].start,wr[2].stop,1)
        eregions += (region_start+(xer,slice(ops,SPLITY+ops,1),xer,slice(sy,sy+SPLITY,1)),)
    if c1 and c2:
        boundary_regions += (region_start+(slice(ops,SPLITX+tops,1),slice(ops,SPLITY+tops,1),slice(sx,ss[2],1),slice(sy,ss[3],1)),)
        eregions += (region_start+(slice(ops,SPLITX+ops,1),slice(ops,SPLITY+ops,1),slice(sx,sx+SPLITX,1),slice(sy,sy+SPLITY,1)),)
        #A bridge can never be on a corner so there is not bridge communication here
    return boundary_regions,eregions

def create_bridges(wr,SPLITX,SPLITY,ops,ss,bridge_slices,block_size):
    """Use this function to create the forward bridges."""
    x_regions = tuple()
    y_regions = tuple()
    region_start = wr[:2]
    c1 = wr[2].start==ops
    c2 = wr[3].start==ops
    sx = ss[2]-ops-SPLITX
    ox = sx-ops
    sy = ss[3]-ops-SPLITY
    oy = sy-ops
    y_blocks = int((wr[3].stop-wr[3].start)/block_size[1])
    x_blocks = int((wr[2].stop-wr[2].start)/block_size[0])
    if c1: #Top edge -  periodic x
        #Boundaries for up pyramid and octahedron
        for x,y in bridge_slices[1]:
            xtt = tuple()   #x temp tuple
            tfxe = x.stop+ox
            xc = tfxe < ss[2]
            tfxe = tfxe if xc else ss[2]
            tfx = slice(x.start+ox,tfxe,1)
            nx = x if xc else slice(x.start,tfx.stop-tfx.start+x.start,1)
            tfys = wr[3].start-ops+y.start+SPLITY
            tfy = slice(tfys,tfys+(y.stop-y.start),1)
            #Adjustment for multiple blocks per rank
            xtt += ((nx,y,tfx,tfy),)
            for i in range(1,y_blocks):
                y = slice(y.start+i*block_size[1],y.stop+i*block_size[1],1)
                tfy = slice(tfy.start+i*block_size[1],tfy.stop+i*block_size[1],1)
                xtt += ((nx,y,tfx,tfy),)
            x_regions += (xtt,)
    if c2: #Side edge -  periodic y
        for x,y in bridge_slices[0]:
            #Finding forward bridge
            ytt = tuple()   #x temp tuple
            tfye = y.stop+ox
            yc = tfye < ss[3]
            tfye = tfye if yc else ss[3]
            tfy = slice(y.start+oy,tfye,1)
            ny = y if yc else slice(y.start,tfy.stop-tfy.start+y.start,1)
            tfxs = wr[2].start-ops+x.start+SPLITX
            tfx = slice(tfxs,tfxs+(x.stop-x.start),1)
            ytt += ((x,ny,tfx,tfy),)
            #Adjustment for multiple blocks
            for i in range(1,x_blocks):
                x = slice(x.start+i*block_size[0],x.stop+i*block_size[0],1)
                tfx = slice(tfx.start+i*block_size[0],tfx.stop+i*block_size[0],1)
                ytt += ((x,ny,tfx,tfy),)
            y_regions += (ytt,)
    return x_regions,y_regions

def create_rev_bridges(wr,SPLITX,SPLITY,ops,ss,bridge_slices,block_size):
    """Use this function to create boundary write regions."""
    x_regions = tuple()
    y_regions = tuple()
    sx = ss[2]-ops-SPLITX
    sy = ss[3]-ops-SPLITY
    region_start = wr[:2]
    c1 = wr[2].stop==sx
    c2 = wr[3].stop==sy
    tops = 2*ops
    y_blocks = int((wr[3].stop-wr[3].start)/block_size[1])
    x_blocks = int((wr[2].stop-wr[2].start)/block_size[0])
    if c1: #Top edge -  periodic x
        for x,y in bridge_slices[1]:
            xtt = tuple()
            xc = x.start > SPLITX
            nx = x if xc else slice(SPLITX,x.stop,1)
            tfxs = 0+nx.start-SPLITX
            tfx = slice(tfxs,tfxs+(nx.stop-nx.start),1)
            tfys = wr[3].start-ops+y.start
            tfy = slice(tfys,tfys+(y.stop-y.start),1)
            #Adjustment for multiple blocks per rank
            xtt += ((nx,y,tfx,tfy),)
            for i in range(1,y_blocks):
                y = slice(y.start+i*block_size[1],y.stop+i*block_size[1],1)
                tfy = slice(tfy.start+i*block_size[1],tfy.stop+i*block_size[1],1)
                xtt += ((nx,y,tfx,tfy),)
            x_regions += (xtt,)
    if c2: #Side edge -  periodic y
        for x,y in bridge_slices[0]:
            ytt = tuple()
            yc = y.start > SPLITY
            ny = y if yc else slice(SPLITY,y.stop,1)
            tfys = 0+ny.start-SPLITY
            tfy = slice(tfys,tfys+(ny.stop-ny.start),1)
            tfxs = wr[2].start-ops+x.start
            tfx = slice(tfxs,tfxs+(x.stop-x.start),1)
            ytt += ((x,ny,tfx,tfy),)
            #Adjustment for multiple blocks
            for i in range(1,x_blocks):
                x = slice(x.start+i*block_size[0],x.stop+i*block_size[0],1)
                tfx = slice(tfx.start+i*block_size[0],tfx.stop+i*block_size[0],1)
                ytt += ((x,ny,tfx,tfy),)
            y_regions += (ytt,)
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

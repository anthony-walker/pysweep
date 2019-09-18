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

def create_boundaries(wr,SPLITX,SPLITY,ops,ss,bridge_slices):
    """Use this function to create boundary write regions."""
    boundary_regions = tuple()
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
    if c2: #Side edge -  periodic y
        boundary_regions += (region_start+(x_reg,slice(ops,SPLITY+tops,1),wr[2],slice(sy,ss[3],1)),)
    if c1 and c2:
        boundary_regions += (region_start+(slice(ops,SPLITX+tops,1),slice(ops,SPLITY+tops,1),slice(sx,ss[2],1),slice(sy,ss[3],1)),)
    return boundary_regions

def create_shifted_boundaries(wr,SPLITX,SPLITY,ops,ss,bs):
    """Use this function to create boundary write regions."""
    shifted_boundary_regions = tuple()
    region_start = wr[:2]
    c1 = wr[2].stop==ss[2]-ops
    c2 = wr[3].stop==ss[3]-ops
    xst = wr[2].stop-wr[2].start
    yst = wr[3].stop-wr[3].start
    x_reg = slice(ops,xst+ops,1)
    y_reg = slice(ops,yst+ops,1)
    tops = 2*ops
    if c1: #Bottom edge -  periodic x
        #Boundaries for up pyramid and octahedron
        shifted_boundary_regions += (region_start+(slice(xst-SPLITX,xst+ops,1),y_reg,slice(0,SPLITX+ops,1),wr[3]),)
    if c2: #Right edge -  periodic y
        shifted_boundary_regions += (region_start+(x_reg,slice(yst-SPLITY,yst+ops,1),wr[2],slice(0,SPLITY+ops,1)),)
    if c1 and c2:
        shifted_boundary_regions += (region_start+(slice(xst-SPLITX,xst+ops,1),slice(yst-SPLITY,yst+ops,1),slice(0,SPLITX+ops,1),slice(0,SPLITY+ops,1)),)
    return shifted_boundary_regions

def create_standard_bridges(XR,ops,bgs,ss,bs,rank=None):
    """Creating reg bridge write tuples"""
    pwxt = tuple()
    wxt = tuple()
    sx = int(ss[2]-bs[0]/2-2*ops)
    sy = int(ss[3]-bs[1]/2-2*ops)
    # wyt = tuple()
    c1 = XR[2].start==0
    c2 = XR[3].start==0
    x_blocks = int((XR[2].stop-XR[2].start-2*ops)/bs[0])
    y_blocks = int((XR[3].stop-XR[3].start-2*ops)/bs[1])
    #Standard bridge writes
    for x,y in bgs:
        wxt = tuple()
        ttt = tuple()
        #X-write
        wxs = x.start+XR[2].start
        wxst = wxs+x.stop-x.start
        #Y-write
        wys = y.start+XR[3].start
        wyst = wys+y.stop-y.start
        for i in range(x_blocks):
            for j in range(y_blocks):
                xsl = slice(x.start+bs[0]*i,x.stop+bs[0]*i,1)
                ysl = slice(y.start+bs[1]*j,y.stop+bs[1]*j,1)
                sxsl = slice(wxs+bs[0]*i,wxst+bs[0]*i,1)
                sysl = slice(wys+bs[1]*j,wyst+bs[1]*j,1)
                ttt += ((xsl,ysl,sxsl,sysl),)
        wxt += ttt,
        #Edge Bridge Writes
        if c1: #Top edge -  periodic x
            #X-write
            xtt = tuple()
            tfxe = x.stop+sx
            xc = tfxe < ss[2]
            tfxe = tfxe if xc else ss[2]
            tfx = slice(x.start+sx,tfxe,1)
            nx = x if xc else slice(x.start,tfx.stop-tfx.start+x.start,1)
            #Y-write
            wys = y.start+XR[3].start
            wyst = wys+y.stop-y.start
            tfy = slice(wys,wyst,1)
            xtt += ((nx,y,tfx,tfy),)
            for i in range(1,y_blocks):
                y = slice(y.start+bs[1],y.stop+bs[1],1)
                tfy = slice(tfy.start+bs[1],tfy.stop+bs[1],1)
                xtt += ((nx,y,tfx,tfy),)
            wxt += (xtt,)
        if c2: #Side edge -  periodic y
            #X-write
            ytt = tuple()
            wxs = x.start+XR[2].start
            wxst = wxs+x.stop-x.start
            tfx = slice(wxs,wxst,1)
            #Y-write
            tfye = y.stop+sy
            yc = tfye < ss[3]
            tfye = tfye if yc else ss[3]
            tfy = slice(y.start+sy,tfye,1)
            ny = y if yc else slice(y.start,tfy.stop-tfy.start+y.start,1)
            ytt += ((x,ny,tfx,tfy),)
            #Adjustment for multiple blocks
            for i in range(1,x_blocks):
                x = slice(x.start+bs[0],x.stop+bs[0],1)
                tfx = slice(tfx.start+bs[0],tfx.stop+bs[0],1)
                ytt += ((x,ny,tfx,tfy),)
            wxt += (ytt,)
        if wxt:
            pwxt += wxt,
    return pwxt

def create_shifted_bridges(XR,ops,bgs,ss,bs,rank=None):
    """Creating reg bridge write tuples"""
    pwxt = tuple()
    wxt = tuple()
    SPX = int(bs[0]/2)
    SPY = int(bs[1]/2)
    sx = int(ss[2]-SPX-2*ops)
    sy = int(ss[3]-SPY-2*ops)
    c1 = XR[2].stop==ss[2]
    c2 = XR[3].stop==ss[3]
    x_blocks = int((XR[2].stop-XR[2].start-2*ops)/bs[0])
    y_blocks = int((XR[3].stop-XR[3].start-2*ops)/bs[1])
    #Standard bridge writes
    for x,y in bgs:
        wxt = tuple()
        ttt = tuple()
        #X-write
        wxs = x.start+XR[2].start
        wxst = wxs+x.stop-x.start
        #Y-write
        wys = y.start+XR[3].start
        wyst = wys+y.stop-y.start
        for i in range(x_blocks):
            for j in range(y_blocks):
                xsl = slice(x.start+bs[0]*i,x.stop+bs[0]*i,1)
                ysl = slice(y.start+bs[1]*j,y.stop+bs[1]*j,1)
                sxsl = slice(wxs+bs[0]*i,wxst+bs[0]*i,1)
                sysl = slice(wys+bs[1]*j,wyst+bs[1]*j,1)
                ttt += ((xsl,ysl,sxsl,sysl),)
        wxt += ttt,
        #Edge Bridge Writes
        if c1: #Top edge -  periodic x
            xtt = tuple()
            xc = x.start > SPX
            nx = x if xc else slice(SPX,x.stop,1)
            tfxs = 0+nx.start-SPX
            tfx = slice(tfxs,tfxs+(nx.stop-nx.start),1)
            tfys = XR[3].start+y.start
            tfy = slice(tfys,tfys+(y.stop-y.start),1)
            #Adjustment for multiple blocks per rank
            xtt += ((nx,y,tfx,tfy),)
            for i in range(1,y_blocks):
                y = slice(y.start+bs[1],y.stop+bs[1],1)
                tfy = slice(tfy.start+bs[1],tfy.stop+bs[1],1)
                xtt += ((nx,y,tfx,tfy),)
            wxt += (xtt,)
        if c2: #Side edge -  periodic y
            ytt = tuple()
            yc = y.start > SPY
            ny = y if yc else slice(SPY,y.stop,1)
            tfys = 0+ny.start-SPY
            tfy = slice(tfys,tfys+(ny.stop-ny.start),1)
            tfxs = XR[2].start+x.start
            tfx = slice(tfxs,tfxs+(x.stop-x.start),1)
            ytt += ((x,ny,tfx,tfy),)
            #Adjustment for multiple blocks
            for i in range(1,x_blocks):
                x = slice(x.start+bs[0],x.stop+bs[0],1)
                tfx = slice(tfx.start+bs[0],tfx.stop+bs[0],1)
                ytt += ((x,ny,tfx,tfy),)
            wxt += (ytt,)
        if wxt:
            pwxt += wxt,
    return pwxt

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

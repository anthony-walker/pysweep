#Programmer: Anthony Walker
#THis file contains all the necessary functions to create blocks and cpu sets for
#the swept rule.
import numpy as np

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

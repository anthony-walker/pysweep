#Programmer: Anthony Walker
#THis file contains all the necessary functions to create blocks and cpu sets for
#the swept rule.
from itertools import product
import numpy as np

def create_dist_up_sets(block_size,ops):
    """This function creates up sets for the dsweep."""
    sets = tuple()
    ul = block_size[0]-ops
    dl = ops
    while ul > dl:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl+=ops
        ul-=ops
    return sets

def create_dist_down_sets(block_size,ops):
    """Use this function to create the down pyramid sets from up sets."""
    bsx = int(block_size[0]/2)
    bsy = int(block_size[1]/2)
    dl = int((bsy)-ops); #lower y
    ul = int((bsy)+ops); #upper y
    sets = tuple()
    while dl > 0:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl-=ops
        ul+=ops
    return sets

def create_dist_bridge_sets(block_size,ops,MPSS):
    """Use this function to create the iidx sets for bridges."""
    sets = tuple()
    xul = block_size[0]-ops
    xdl = ops
    yul = int(block_size[0]/2+ops)
    ydl = int(block_size[0]/2-ops)
    xts = xul
    xbs = xdl
    for i in range(MPSS):
        sets+=(tuple(product(np.arange(xdl,xul,1),np.arange(ydl,yul,1))),)
        xdl+=ops
        xul-=ops
        ydl-=ops
        yul+=ops
    return sets,sets[::-1]

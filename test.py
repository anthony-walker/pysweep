
import numpy as np
from src.equations.eqt import *
import sys

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.0f"%si+", ")
        sys.stdout.write("]\n")

v = 4*5*6*8
# for i in range(6,34,2):
#     print(i,v,(v)%i)

x = 10
arr = np.zeros((10,4,x,x))
patt = np.zeros((x+1))

for i in range(1,x+1,2):
    patt[i] = 1
sb = True
for i in range(4):
    for j in range(x):
        if sb:
            arr[0,i,j,:] = patt[0:-1]
        else:
            arr[0,i,j,:] = patt[1:]
        sb = not sb


pm(arr,0)
iidx = tuple()
for i in range(1,x-1):
    for j in range(1,x-1):
        iidx+=(i,j),

step(arr,iidx,0,1)
pm(arr,1)

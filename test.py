import numpy as np
from src.equations import euler
from src.equations import euler1D
from src.analytical import *
import sys

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.1e"%si+", ")
        sys.stdout.write("]\n")

def arr_update(i,arr0,ops):
    arr0[i,:,:ops,ops:x-ops] = arr0[i,:,-2*ops:-ops,ops:x-ops]
    arr0[i,:,ops:x-ops,:ops] = arr0[i,:,ops:x-ops,-2*ops:-ops]
    arr0[i,:,-ops:,ops:x-ops] = arr0[i,:,ops:2*ops,ops:x-ops]
    arr0[i,:,ops:x-ops,-ops:] = arr0[i,:,ops:x-ops,ops:2*ops]

tf = 1
t0 = 0
dt = 0.001
t = int((tf-t0)/dt)
x = 14
v = 4
gamma = 1.4
ops = 2
X=Y=1
arr0 = np.zeros((t,v,x,x))
cvics = vics()
print(cvics.Shu(gamma,10).state)


#Dimensions and steps
dx = X/(x-2*ops)
dy = Y/(x-2*ops)
iidx = tuple()
for i in range(ops,x-ops):
    for j in range(ops,x-ops):
        iidx += (i,j),
#Creating initial vortex from analytical code

# av = vortex(cvics,X,Y,x-2*ops,x-2*ops,times=np.arange(t0,tf,dt))
# av = convert_to_flux(av,gamma)
# flux_vortex = convert_to_flux(initial_vortex,gamma)[0]
# arr0[0,:,ops:-ops,ops:-ops] = flux_vortex[:,:,:]
# arr_update(0,arr0,ops)
# euler.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
# #Changing arguments
# for i in range(t-1):
#     euler.step(arr0,iidx,i,i)
#     arr_update(i+1,arr0,ops)
#     print(np.amax(arr0[i+1,:,ops:x-2,ops:x-2]-av[i+1,:,:,:]))
#     input()

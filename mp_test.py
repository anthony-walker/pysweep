import numpy as np
import multiprocessing as mp
import ctypes
from mpi4py import MPI
import os
# sarr = None
#
# def main_test():
#     global sarr
#     shared_shape = (5,5,5)
#     sarr_base = mp.Array(ctypes.c_float, int(np.prod(shared_shape)),lock=False)
#     sarr = np.ctypeslib.as_array(sarr_base)
#     sarr = sarr.reshape(shared_shape)
#     ppool = mp.Pool()
#     for i in range(shared_shape[0]):
#         sarr[i,:,:] = 0
#     ppool.map(function_test,range(shared_shape[0]))
#     print(sarr)
#
# def function_test(val):
#     sarr[val,:,:] = val
#
# main_test()
nx = ny = 12*13
BS = (12,12,1)
t0 = 0
tf = 0.1
dt = 0.01
dx = dy = 0.1
gamma = 1.4
aff = 0.5
arr = np.ones((4,nx,ny))
ranks = np.arange(0,6,1)
dGdx = [2,1,0,0,0,0]
tnc = len(ranks)
tng = np.sum(dGdx)

NB = np.prod(arr.shape[1:])/np.prod(BS)
assert (NB).is_integer(), "Provided array dimensions is not divisible by the specified block size."
NR = arr.shape[1]/BS[0]
GNR = np.ceil(NR*aff)
CNR = NR-GNR
k = (GNR-GNR%tng)/tng
n = GNR%tng
m = tng-n
# print((k+1)*n+k*m,GNR)
assert (k+1)*n+k*m == GNR
kc = (CNR-CNR%tnc)/tnc
nc = CNR%tnc
mc = tnc-nc
print((kc+1)*nc+kc*mc,CNR)
print(kc,nc,mc)

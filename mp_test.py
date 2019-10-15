import numpy as np
import multiprocessing as mp
import ctypes

sarr = None

def main_test():
    global sarr
    shared_shape = (5,5,5)
    sarr_base = mp.Array(ctypes.c_float, int(np.prod(shared_shape)),lock=False)
    sarr = np.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    ppool = mp.Pool()
    for i in range(shared_shape[0]):
        sarr[i,:,:] = 0
    ppool.map(function_test,range(shared_shape[0]))
    print(sarr)

def function_test(val):
    sarr[val,:,:] = val

main_test()

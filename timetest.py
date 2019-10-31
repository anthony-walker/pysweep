from itertools import cycle
from time import time

def A():
    ct = 0
    for i in range(100):
        ct += i


def B():
    ct = 0
    for i in range(100):
        ct += i

def tf1():
    mc = cycle([A,B])
    for i in range(40):
        next(mc)()

def tf2():
    for i in range(20):
        A()
        B()

test = 1000000
start = time()
for i in range(test):
    tf1()
stop = time()
print("TF1 "+str(stop-start))

start = time()
for i in range(test):
    tf2()
stop = time()
print("TF2 "+str(stop-start))

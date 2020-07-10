# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:46:11 2020

@author: 1052668570
"""

from numba import jit
import timeit
import numpy as np
import pandas as pd
import time


class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def numba_suma(self, j):
        self.suma(self.a, self.b, j)

    @staticmethod
    @jit(nopython=True)
    def suma(a, b, j):
        res = 0
        for _ in range(j):
            res += (a + b)
        return res

    def normal_suma(self, j):
        res = 0
        a = self.a
        b = self.b
        for _ in range(j):
            res += (a + b)**2
            for _ in range(10):
                res += (a+ b)

        return res


c1 = A(2, 2)
start_time = time.time()
c1.numba_suma(100000)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
c1.normal_suma(100000)
print("--- %s seconds ---" % (time.time() - start_time))


# a = np.random.randn(100)
# a.shape

def batch(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment


for i in batch(0, 86400, 865):
    print(i)
    
vals = np.array([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [])
cols = ['a', 'b', 'c']

data = {var: val for var, val in zip(cols, vals)}
   
    
    
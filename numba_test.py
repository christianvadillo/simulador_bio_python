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


# class A:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

#     def numba_suma(self, j):
#         self.suma(self.a, self.b, j)

#     @staticmethod
#     @jit(nopython=True)
#     def suma(a, b, j):
#         res = 0
#         for _ in range(j):
#             res += (a + b)
#         return res

#     def normal_suma(self, j):
#         res = 0
#         for _ in range(j):
#             res += (self.a + self.b)
#         return res


# c1 = A(2, 2)
# start_time = time.time()
# c1.numba_suma(10000000)
# print("--- %s seconds ---" % (time.time() - start_time))


a = np.random.randn(100)
a.shape

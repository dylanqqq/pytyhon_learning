# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:46:30 2021

@author: 钟牙
"""
import matplotlib.pyplot as plt

import numpy as np
n =5
mat= np.zeros((n,n))

mat[0,:] = np.arange(1,n+1,1)
for i in range(1,n):
    mat[i,n-1] = mat[i-1,0]
    for j in range(n-1):
        mat[i,j] = mat[i-1,j+1]



for i in range(n):
    for j in range(n):
        mat[i,j] = (i+j)% n +1
#2

n=6
res = 1
for i in range(1, n+1):
    res = res* i
    print(res)
    
def factorial():
    res = n
    for i in range(1,n):
        res *= i
        return res
print(factorial(4))


#3
from scipy.optimize import minimize

def f(x):
    return x**4 - x**2
print(f(2))
sol = minimize(f,x0 =2).x
print(sol)

plt.plot(sol)
#4


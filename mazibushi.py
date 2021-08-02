# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
a = 1
b = 0.5
c = a + b
d = a * b
e = True
f = False
e * e
f * f
e * f

arr = np.array([1,2,3])
arr2= 2 * arr
arr + 1
arr / 2

print("woxiangshuijiao")
print(a)
print(arr)
plt.plot(arr)
plt.plot(arr2)

arr[0]
arr[3]
arr_part = arr[0:2]
arr_part2 = arr[1:3]
arr_zeros = np.zeros(100)
arr_ones = np.ones(100)
arr_twos = arr_ones + 1
arr_lin = np.linspace(0,1,101)
arr_lin2 = np.linspace(1,2,101)

arr_lin3 = arr_lin * arr_lin2
np.sum(arr_lin3)
np.dot(arr_lin,arr_lin2)
arr_lin4 = arr_lin * arr_lin - arr_lin
plt.figure()
plt.plot(arr_lin4)

arr_sq = np.zeros(100)
for i in range(100):
    arr_sq[i] = i + 1
    

arr_flag = np.zeros(100)

for i in range(100):
    if arr_sq[i] < 50:
        arr_flag[i] = 0
    else:
        arr_flag[i] = 1
    
x = 1.1
x**2
np.sin(x)
np.exp(x)

def f(x):
    y = x**2 + np.sin(x) - np.exp(x)
    return y
f(1)

x_arr = np.linspace(-1,1,101)
y_arr = np.zeros(101)

for i in range(101):
    y_arr[i] = f(x_arr[i])
    
plt.plot(x_arr,y_arr)
plt.figure()
plt.scatter(x_arr,y_arr)

y_arr2 = f(x_arr)

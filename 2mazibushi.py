# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:09:33 2021

@author: -
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

simple_size = 11
x_array =np. linspace(0,10,simple_size)
slope =2
intercept = 1
y_array = x_array * slope +intercept

std = 0.9
epsilon = np.random.normal(0,std, simple_size)

y_array_2 =y_array +epsilon
plt.figure()
plt.scatter(x_array,y_array)
plt.figure()
plt.scatter(x_array,y_array_2)

model = LinearRegression()
model.fit(x_array.reshape(simple_size,1),y_array_2)
a =model.coef_
b =model.intercept_
          
z_array = np.linspace(0,1,100)
z_predict = model.predict(z_array.reshape(100,1))
plt.figure()
plt.scatter(z_array,z_predict)


from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y= iris.target
model_iris = LinearRegression()
model_iris.fit(x,y)
z = np.array([[5.6,3,6,2],[4.6,3.1,2,0.5]])
z_predict = model_iris.predict(z)





def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y

x=np.linspace(-3,3,101)
y=sigmoid(x)

plt.plot(y)
plt.figure()
plt.scatter(x,y)


sample_size =100
x = np.random.uniform(0,10,(sample_size,2))
x1=x[:,0]
x2=x[:,1]
plt.scatter(x1,x2)
y = np.zeros(sample_size)

z = np.random.normal(0,0.5,sample_size)
for i in range(sample_size):
    if x2[i] > x1[i] + z[i]:
        y[i] = 1
    else:
        y[i]=0
        
y0_index = (y == 1)
y1_index = (y == 0)
plt.scatter(x1[y0_index],x2[y0_index],color = 'red')
plt.scatter(x1[y1_index],x2[y1_index],color = 'green')


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x,y)
print(model.intercept_)
print(model.coef_)
print(model.score(x, y))
x_test= np.array([[10,0],
                  [0,10],
                  [5,2]
                  ])
y_test_pred = model.predict(x_test)

xt1 = np.arange(-0.1,10.1,0.1)
xt2 = np.arange(-0.1,10.1,0.1)

xxt1, xxt2 = np.meshgrid(xt1, xt2)

xt = np.hstack([xxt1.reshape(-1,1),xxt2.reshape(-1,1)])
yt = model.predict(xt)

plt.scatter(xt[yt==0, 0],xt[yt==0,1],color = 'red')

plt.scatter(xt[yt==1, 0],xt[yt==1,1],color = 'purple')


iris = load_iris()
x = iris.data
y = iris.target
model = LogisticRegression()
model.fit(x,y)
z = np.array([[5.6,3,6,2],[4.6,3.1,2,0.5]])
z_predict = model.predict(z)
model.score(x,y)



a =np.array([1,2,3])
a[0]
b=a.reshape(3,1)
b[2,0]

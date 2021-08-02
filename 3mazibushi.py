# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:06:29 2021

@author: 钟牙
"""
import numpy as np
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
a = [1,2,3]
a =[]
a.append(1)
a.append(10)
b=np.array(a)
c=list(b)


a = np.ones(10)
b = a
b = a.copy()

import pandas as pd
path="C:/Users/钟牙/Desktop/iris.csv"
data = pd.read_csv(path,index_col = 0)
data.columns
data_v=data.values

data.iloc[0,0]# index location
data.iloc[:,0]
data.iloc[:,-1]

model = KNeighborsClassifier(n_neighbors = 10,p=1)
x = data.iloc[:,0:4]
y = data.iloc[:,4]

model.fit(x,y)
model.score(x,y)

area1 = data.iloc[:,0] * data.iloc[:,1]
area2 = data.iloc[:,2] * data.iloc[:,3]
data['area1'] =area1
data['area2'] =area2
data_final = data.iloc[:,[0,1,2,3,5,6,4]]

save_path = 'C:/Users/钟牙/Desktop/iris_new.csv'
data_final.to_csv(save_path)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size =0.4)
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(train_x,train_y)
print(model.score(train_x,train_y))


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*x -2*x -5
def fderiv(x):
    return 2*x -2

learing_rate =0.1
n_iter = 100

xs = np.zeros(n_iter + 1)
xs[0] = 100

for i in range(n_iter):
    xs[i+1] = xs[i] - learing_rate * fderiv(xs[i])    
    
plt.plot(xs)

from scipy.optimize import minimize

a = minimize(f,x0=100),x

def f2(x):
    return np.exp(-x**2) * (x**2)

b = minimize(f2,x0 =2),x



def E(x):
    a = -x*np.log(x) - (1-x)*np.log(1-x)
    return a

x = np.linspace(0.01,0.99,100)
y = E(x)
plt.figure()
plt.plot(x,y)



wm = pd.read_csv('C:/Users/钟牙/Desktop/wm_1.csv',index_col = 0)
x = wm.iloc[:,:6]
y = wm.iloc[:,6]
import sklearn.tree as tree
model = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 3)
model.fit(x,y)
plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True)
tree.plot_tree(model,filled = True)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
bc = load_breast_cancer()
x = bc.data
y = bc.target
x_train,x_test,y_train,y_test = \
    train_test_split(x,y,train_size=0.6)
model = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))






from sklearn.neighbors import  KNeighborsClassifier

model_1=KNeighborsClassifier()
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size =0.4)
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(train_x,train_y)
print(model.score(train_x,train_y))




from sklearn.linear_model import LogisticRegression


sample_size =1000
tmp = np.random.uniform(-10,10,(sample_size,2))
plt.scatter(tmp[:,0],tmp[:,1])
radius = 7
label = np.zeros(sample_size)
for i in range(sample_size):
    if np.sqrt(tmp[i,0]**2+tmp[i,1]**2) < radius:label[i]=1
    
plt.figure(figsize= (10,10))
plt.scatter(tmp[label==0,0],tmp[label==0,1],color = 'g')
plt.scatter(tmp[label==1,0],tmp[label==1,1],color = 'b')

model = LogisticRegression()
model.fit(x,label)
model.score(x,label)






from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


sample_size = 10000
X = np.random.uniform(-10, 10, (sample_size, 2))
plt.scatter(X[:, 0], X[:, 1])
radius = 8
labels = np.zeros(sample_size)

for i in range(sample_size):
    if np.sqrt(X[i, 0] ** 2 ++ X[i, 1] ** 2) <= radius:
        labels[i] = 1
    else:
        labels[i] = 0
        
plt.figure(figsize=(10, 10))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1])
plt.scatter(X[labels == 1, 0], X[labels == 1, 1])

#logistic
model = LogisticRegression()
model.fit(X,labels)

x_test = np.random.uniform(-10,10,(10000,2))
y_pred = model.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred == 0, 0], x_test[y_pred == 0, 1])
plt.scatter(x_test[y_pred == 1, 0], x_test[y_pred == 1, 1])



#KNN
model2 = KNeighborsClassifier()
model2.fit(X,labels)

x_test = np.random.uniform(-10,10,(10000,2))
y_pred = model2.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred == 0, 0], x_test[y_pred == 0, 1])
plt.scatter(x_test[y_pred == 1, 0], x_test[y_pred == 1, 1])


#决策树
import sklearn.tree as tree
model3 = tree.DecisionTreeClassifier(criterion='entropy')
model3.fit(X,labels)

x_test = np.random.uniform(-10,10,(10000,2))
y_pred = model3.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred == 0, 0], x_test[y_pred == 0, 1])
plt.scatter(x_test[y_pred == 1, 0], x_test[y_pred == 1, 1])

#神经网络
from sklearn.neural_network import MLPRegressor
hidden_layer_sizes= (200, )
model4 = MLPRegressor(hidden_layer_sizes)
model4.fit(X,labels)

x_test = np.random.uniform(-10,10,(10000,2))
y_pred = model3.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred == 0, 0], x_test[y_pred == 0, 1])
plt.scatter(x_test[y_pred == 1, 0], x_test[y_pred == 1, 1])

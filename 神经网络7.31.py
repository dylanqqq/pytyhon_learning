# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:24:53 2021

@author: 钟牙
"""

import numpy as np
import pandas as pd

data = np.random.randn(5,3)
df = pd.DataFrame(data,columns=['one','two','three'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

#使用0填充空值
df.fillna(0,inplace = True)
df_1 = df. fillna(0)

#使用ffill填充空值
df_2 = df.fillna(method= 'ffill')
df_3 = df_2.fillna(0)


#使用平均值进行填充
mean = df['three'].mean()
df['three'].fillna(mean,inplace=True)
#或者
#df.iloc[:,1].fillna(mean,inplace=True)


#%% neural network
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
sample_size = 10000
x = np.linspace(0,6,sample_size)
y = np.sin(x)

plt.show()
plt.scatter(x,y)

hidden_layer_sizes= (2000, )
model = MLPRegressor(hidden_layer_sizes)
model.fit(x.reshape((sample_size,1)),y)


z_predict = model.predict(x.reshape((sample_size,1)))
plt.show()
plt.scatter(x,y)
plt.scatter(x,z_predict)
plt.title(str(hidden_layer_sizes))


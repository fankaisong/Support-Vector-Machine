#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

from SVM import *   
                
        



df = pd.read_csv('iris.csv')

sepalLength = df['SepalLengthCm'][:100].values
sepalWidth = df['SepalWidthCm'][:100].values
species = df['Species'][:100].values

feature = []
for i in range(100):
    temp = []
    temp = [sepalLength[i]] + [sepalWidth[i]]
    feature.append(temp)

#Iris-setosa is 1   Iris-versicolor is -1
label = [1]*50 + [-1]*50


#shuffle
feature, label = shuffle(feature,label)
#split dataset into 90% for training and 10% validation
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.1)

feature_train = np.array(feature_train)
label_train = np.array(label_train)

feature_test = np.array(feature_test)
label_test = np.array(label_test)


# In[208]:


a = 0.01
epoch = 8000
svm = SVM()
svm.train(feature_train, label_train,epoch, a)

res,acc = svm.predict(feature_test,label_test)

print(acc)







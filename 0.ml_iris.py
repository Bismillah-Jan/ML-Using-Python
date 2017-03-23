#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:35:04 2017

@author: bismillah
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
iris=load_iris()
#print (type(iris))
#print (len(iris.data))
#print (iris.feature_names)
#print (iris.target)
#print (iris.target_names)

print (iris.data.shape, iris.target.shape)
X=iris.data
y=iris.target

knn=KNeighborsClassifier(n_neighbors=1)
#now we have an object knn which knows how to do KNeighbour classification

#print (knn)

knn.fit(X,y)

x_new=np.array( [[3,5,4,2],[5,4,3,2]])
print (knn.predict(x_new))

logReg=LogisticRegression()
print (logReg)
logReg.fit(X,y)
print (logReg.predict(x_new))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:07:44 2017

@author: bismillah
"""
import scipy
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
filename = "breast-cancer-wisconsin.data"
inFile=open(filename, 'r')
data=inFile.read()

data=pd.read_csv(filename)
data = scipy.array(data)

#remove patients ID
X = data[:, 1:10]
y = data[:, 10] #extract labels

xTrain, xTest, yTrain, yTest=train_test_split(X, y, test_size=0.4)

print (xTrain.shape)
print (xTest.shape)

kRange=range(1, 70)
accList=[]
#plot accuracy vs k in the following few lines
for i in kRange:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xTrain, yTrain) 
    yPred1=knn.predict(xTest)
    knnAcc=metrics.accuracy_score(yTest, yPred1)
    accList.append(knnAcc)

plt.plot(kRange,accList)    
print "Ks: ",kRange
print "Corresponding Accuracy: ", accList
#plz uncomment the following lines to use svm as classifier
""" 
svmclf=svm.SVC()
svmclf.fit(xTrain, yTrain) 
yPred2=svmclf.predict(xTest)
svmAcc=metrics.accuracy_score(yTest, yPred2)
print ("svm acc: ", svmAcc)
"""
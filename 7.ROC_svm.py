#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 01:21:00 2017

@author: bismillah
"""

import scipy
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale,PolynomialFeatures, label_binarize
col_names=['pregnant', 'glucose', 'bp', 'skin',  'insulin', 'bmi', 'pidegree', 'age', 'label']
pima=pd.read_csv("pima_indian_diabetes.csv", header=None, names=col_names )
pima.head()


# In[3]:

feature_col=['pregnant','glucose','bp','insulin']
X=pima[feature_col]
X=scale(X)
y=pima.label



xtr,xtest,ytr,ytest=train_test_split(X,y, test_size=0.7) 

model=svm.SVC(C=1000, kernel='rbf', degree=3, probability=True)
model.fit(xtr,ytr)
yPred=model.predict(xtest)
acc=metrics.accuracy_score(ytest, yPred)

print acc
null_acc= max(ytest.mean(), 1-ytest.mean())
print null_acc
print "True: ", ytest[0:25]
print "Pred: ", yPred[0:25]



confusion=metrics.confusion_matrix(ytest, yPred)
print confusion

TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]


Acc=(TP+TN)/float(TP+TN+FP+FN)
Miss_cl_rate= (FP+FN)/float(TP+TN+FP+FN)
TPR=TP/float(TP+FN)
TNR= TN/float(TN+FP)
FPR=FP/float(FP+TN)
FNR=FN/float(FN+TP)
Precession= TP/float(TP+FP)

print "TP:",TP , "  FP:",FP,  "  TN:",TN,  "  FN:",FN
print "Accuracy: ", Acc, "vs", acc               #where acc is calculated using metrics procedure
print "Mis-classification Rate: ", Miss_cl_rate  #  also equal to 1-Acc
print "TPR: ", TPR                 #metrics.recall_score(ytest, yPred)
print "TNR: ", TNR                 #specificity=1-FPR
print "FPR: ", FPR 
print "FNR: ", FNR                 #FNR=1-TPR
print "Precession: ", Precession                 #metrics.precision_score(ytest, yPred)


# store the predicted probabilities for class 1
y_pred_prob = model.predict_proba(xtest)[:, 1]


fpr, tpr, thresholds= metrics.roc_curve(ytest, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC for PIMA diabetes dataset")
plt.grid(True)

def evaluate_threshold(threshold):
    print( 'Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


#evaluate_threshold(0.5)

#print(metrics.roc_auc_score(ytest, yPred))

#cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean()



# coding: utf-8

# # Author: Bismillah Jan
# ## Evaluation of Classifier
# * Using Different metrices (Accurac, Confusion Matrix, ROC etc)
# * Using PIMA_Indian_diabetes dataset

# In[1]:

import pandas as pd


# In[2]:

col_names=['pregnant', 'glucose', 'bp', 'skin',  'insulin', 'bmi', 'pidegree', 'age', 'label']
pima=pd.read_csv("pima_indian_diabetes.csv", header=None, names=col_names )
pima.head()


# In[3]:

feature_col=['pregnant','insulin','bmi', 'age']
X=pima[feature_col]
y=pima.label
X.head()


# ## Using train_test_split procedure
# ### Accuracy measure

# In[20]:

from sklearn.cross_validation import train_test_split
from sklearn import metrics
xtr,xtest,ytr,ytest=train_test_split(X,y, test_size=0.3) 


# ### Logistic Regression

# In[21]:

# LogisticRegression
from sklearn.linear_model import LogisticRegression as lg
model=lg()
model.fit(xtr, ytr)
yPred=model.predict(xtest)
acc=metrics.accuracy_score(ytest, yPred)
model.fit(xtr,ytr)


# In[22]:

print acc


# ## Null accuracy
# * what is the percentage of maximum class

# In[23]:

null_acc= max(ytest.mean(), 1-ytest.mean())
print null_acc
print "True: ", ytest.values[0:25]
print "Pred: ", yPred[0:25]


# # Confusion matrix

# In[24]:

confusion=metrics.confusion_matrix(ytest, yPred)
print confusion


# In[25]:

TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]


# In[26]:

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


# ## ROC Curve

# In[27]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[28]:

fpr, tpr, thresholds= metrics.roc_curve(ytest, yPred)
plt.plot(fpr, tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC for PIMA Diabetes Dataset")
plt.grid(True)


# In[29]:

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[30]:

evaluate_threshold(0.5)


# In[33]:

print(metrics.roc_auc_score(ytest, yPred))


# In[35]:

from sklearn.cross_validation import cross_val_score
cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean()


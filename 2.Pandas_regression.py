#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 07:34:58 2017

@author: bismillah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 07:26:58 2017
@author: bismillah
Dataset: Wisconsin Dataset of Breast Cancer
"""

import pandas as pd
import scipy
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

filename = "Advertising.csv"
inFile=open(filename, 'r')
data=inFile.read()
data = pd.read_csv(filename, index_col=0)


#sns.pairplot(data, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.5, kind='reg')

features_cols=['TV', 'Radio','Newspaper']
X=data[features_cols]
y=data['Sales']

print X
xTrain, xTest, yTrain, yTest=train_test_split(X, y, random_state=1)


lineReg=LinearRegression()
lineReg.fit(xTrain,  yTrain)
print lineReg.coef_
print lineReg.intercept_

#zipp=zip(features_cols, lineReg.coef_)
#print (zipp)



y_pred=lineReg.predict(xTest)

print (metrics.mean_absolute_error(yTest , y_pred))










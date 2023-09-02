# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:09:39 2023

@author:
"""


#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt



#READING THE DATA FROM YOUR FILES
data = pd.read_csv('advertising.csv')
data.head()


# TO VISUALISE DATA
fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=[14,7])
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])




#creating x&y for Linear Regression
feature_cols = ['TV']
x = data[feature_cols]
y = data.Sales



#IMPORT LINEAR REGRESSION ALGO FOR SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)




result = 69.7+0.0554*50
print(result)


#CREATE A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()



preds = lr.predict(x_new)
preds


data.plot(kind = 'scatter',x='TV',y='Sales')





import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
lm.conf_int()



#FINDING THE PROBABILITY VALUES
lm.pvalues



# FIND THE R-SQUARED VALUES
lm.rsquared



# Multi LinearRegression
feature_cols = ['TV','Radio','Newspaper']
x = data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(x,y)


print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()


lm = smf.ols(formula='Sales ~ TV+Radio',data=data).fit()
lm.conf_int()
lm.summary()
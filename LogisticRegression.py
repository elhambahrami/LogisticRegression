#!/usr/bin/env python
# coding: utf-8

# #### World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using different models logistic regression.<br>Note:In column 'TenYearCHD' 0 means not at risk of hear diseas(No) and 1 means at risk(Yes).

# In[75]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[76]:


Heart = pd.read_csv('~/Downloads/framingham.csv')


# In[77]:


Heart.info()


# In[78]:


Heart.head()


# In[91]:


Heart['TenYearCHD'].value_counts()


# In[79]:


Heart.isnull().sum()


# ##### So we have some missings since the  number of missing values is small compare to the data size and also it is realted to some important health factor that may affect the model and result therefore I decided to remove them.

# In[80]:


Heart = Heart.dropna()


# #### We start with Binary Logistic Regression 

# In[82]:


X = Heart[['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol',
           'sysBP','diaBP','BMI','heartRate','glucose']]
y = Heart[['TenYearCHD']]


# In[83]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)


# In[84]:


logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


# In[89]:


accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[93]:


import numpy as np
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# #### the ROC_AUC score is not that high it may need some feature engineering!

# In[92]:


logreg = LogisticRegression(C=0.001, penalty='l2')
from sklearn.feature_selection import RFE
rfe = RFE(logreg)
fit = rfe.fit(X_train, y_train)
print("top features: %d" %(fit.n_features_))
print("selected features: %s" %(fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# #### The data set is imbalanced but since it is related to health therefore it's not to do boosting to make it balanced also false negetive is very important therefore we tried to keep the dataset as it is. Accuracy seems to be hight enought to predict.

# #### I am not sure how to use Multinomial Logistic Regression for yes/no prediction since there are only two classes of prediction, also for Ordinal logistic regression since in my data set there are only two distrcit value to predict!

# In[ ]:





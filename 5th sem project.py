#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


#Loading the data
data0 = pd.read_csv(r'5.urldata.csv')
data0.head()


# In[7]:


#Familiarizing with Data
#Checking the shape of the dataset
data0.shape


# In[8]:


data0.columns


# In[9]:


data0.info()


# In[10]:


#visualizing the data
data0.hist(bins=50,figsize=(15,15))
plt.show()


# In[11]:


#cleaning the data
data0.describe()


# In[15]:


data0=data0.drop(['Domain'],axis=1).copy()
data0.head()


# In[16]:


#shuffling the rows in the dataset
data0=data0.sample(frac=1).reset_index(drop=True)
data0.head()


# In[24]:


#splitting the data
y= data0['Label']
x=data0.drop('Label',axis=1)
x.shape, y.shape


# In[30]:


#splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2, random_state=12)
x_train.shape , x_test.shape


# In[31]:


from sklearn.metrics import accuracy_score


# In[32]:


#creating holders to store the model performance results
ML_Model=[]
acc_train=[]
acc_test=[]

#function to call for storing the results
def storeResults(model,a,b):
    ML_Model.append(model)
    acc_train.append(round(a,3))
    acc_test.append(round(b,3))


# In[33]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5)
#fit the model
tree.fit(x_train,y_train)


# In[35]:


#predicting the target value from the model for the samples
y_test_tree = tree.predict(x_test)
y_train_tree = tree.predict(x_train)


# In[37]:


#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

#storing the results
storeResults('Decision Tree',acc_train_tree,acc_test_tree)


# In[38]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5)
forest.fit(x_train,y_train)


# In[39]:


#predicting the target value from the model for the samples
y_test_forest = forest.predict(x_test)
y_train_forest = forest.predict(x_train)


# In[41]:


#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

storeResults('Random Forest',acc_test_forest,acc_test_forest)


# In[46]:


#XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
xgb.fit(x_train,y_train)


# In[48]:


#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(x_test)
y_train_xgb = xgb.predict(x_train)


# In[50]:


#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

storeResults('XGBoost',acc_train_xgb,acc_test_xgb)


# In[54]:


#Support Vector Machine
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=1.0,random_state=12)
svm.fit(x_train,y_train)


# In[56]:


#predicting the target value from the model for the samples
y_test_svm = svm.predict(x_test)
y_train_svm = svm.predict(x_train)


# In[58]:


#computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))
storeResults('SVM',acc_train_svm,acc_test_svm)


# In[63]:


#comparing the models
results=pd.DataFrame({'ML Model':ML_Model,'Train Accuracy': acc_train,'Test Accuracy':acc_test})
results.sort_values(by=['Test Accuracy','Train Accuracy'],ascending=False)


# In[ ]:





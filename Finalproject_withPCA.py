#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt 


# In[4]:


import seaborn as seabornInstance 


# In[9]:


from sklearn.model_selection import train_test_split 


# In[10]:


data=pd.read_csv("electricity.csv")


# In[11]:


print(data.head(7))


# In[12]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[19]:


features = ['tau1','tau2','tau3','tau4','g1','g2','g3','g4']


# In[20]:


x = data.loc[:, features].values
y = data.loc[:,['stabf']].values


# In[21]:


#Standardize the data, Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales.
pca = PCA(n_components=2)


# In[22]:


principalComponents = pca.fit_transform(x)


# In[23]:


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[26]:


principalDf.head(5)
data[['stabf']].head()


# In[27]:


finalDf = pd.concat([principalDf, data[['stabf']]], axis = 1)
finalDf.head(5)


# In[28]:


# Visualize 2D Projection
# Use a PCA projection to 2d to visualize the entire data set. 

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

stabff = ['stable', 'unstable']
colors = ['r', 'g']

for stabf, color in zip(stabff,colors):
    indicesToKeep = finalDf['stabf'] == stabf
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(stabff)
ax.grid()


# In[29]:


principalDf.head(5)


# In[30]:


# Explained Variance
# The explained variance tells us how much information (variance) can be attributed to each of the principal components.

pca.explained_variance_ratio_


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalDf, data[['stabf']], test_size = 0.1, random_state = 0)


# In[33]:


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[34]:


y_pred = classifier.predict(X_test)


# In[35]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[36]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[380]:


# Create your knn classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 12, metric='minkowski', p=2)
classifier.fit(X_train, y_train)


# In[381]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[382]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[383]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[384]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 60, criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[385]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[386]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[387]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[388]:


# Create your classifier here
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train) 


# In[389]:


y_pred = classifier.predict(X_test)


# In[390]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[391]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[392]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# In[393]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[394]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[395]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


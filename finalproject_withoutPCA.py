#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import libraries 
import pandas as pd
import numpy as np


# In[5]:


import matplotlib.pyplot as plt 


# In[6]:


from sklearn.model_selection import train_test_split 


# In[7]:


data=pd.read_csv("electricity.csv")


# In[8]:


#top 10 features selection
import pandas as pd
import numpy as np
data = pd.read_csv("electricity.csv")
X = data.iloc[:,0:12]  #independent columns
y = data.iloc[:,-1]    #target column 
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# In[9]:


#dimensions of the dataset 
data.shape


# In[10]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[11]:


features = ['tau1','tau2','tau3','p1','p2','p4','g1','g3','g4','stab']


# In[12]:


x = data.loc[:, features].values
y = data.loc[:,['stabf']].values


# In[13]:


#standardization of x
x = StandardScaler().fit_transform(x)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 0)


# In[16]:


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[19]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[20]:


# Create your knn classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 12, metric='minkowski', p=2)
classifier.fit(X_train, y_train)


# In[21]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[22]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[23]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 60, criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[25]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[26]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[27]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[29]:


# Create your classifier here
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train, y_train) 


# In[30]:


y_pred = classifier.predict(X_test)


# In[31]:


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[32]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)


# In[33]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# In[34]:


# Predicting the Test set results
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


# In[ ]:





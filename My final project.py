#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[4]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[15]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[17]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)


# In[18]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)


# In[ ]:





# In[21]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[22]:


df[['Principal','terms','age','Gender','education']].head()


# In[23]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[24]:


X = Feature
X[0:5]


# In[25]:


y = df['loan_status'].values
y[0:5]


# In[13]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


df


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


X_train


# In[31]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[32]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[33]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[34]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[35]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[36]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# # The best accuracy was with k=7, ACCURACY = 0.785 

# In[39]:


# K = 7 PRODUCES HIGHEST ACCURACY! :)
from sklearn.tree import DecisionTreeClassifier


# In[40]:


decisionTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
decisionTree # it shows the default parameters


# In[41]:


decisionTree.fit(X_train,y_train)


# In[63]:


predTree = decisionTree.predict(X_test)


# In[64]:


print (predTree [0:5])
print (y_test [0:5])


# In[65]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# # DECISION TREE ACCURACY = 0.61

# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[67]:


yhat = LR.predict(X_test)
yhat


# In[68]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[69]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# # JACCARD SIMILIRATY SCORE FOR LOGISTIC REGRESSION - 0.78 , LOG LOSS 0.54

# In[70]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[71]:


yhat = clf.predict(X_test)
yhat [0:5]


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[73]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# # F1 SCORE - 0.69

# In[74]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[75]:


print ('Kmeans accuracy - 0.78')
print('SVC accuracy - 0.72')
print('Jaccard score - 0.74')
print('Decision tree accuracy - 0.61')


# In[77]:


yhatKNN=neigh.predict(X)
KNNJaccard = jaccard_similarity_score(y, yhatKNN)
KNNF1 = f1_score(y, yhatKNN, average='weighted')
print("Avg F1-score: %.2f" % KNNF1 )
print("KNN Jaccard Score: %.2f" % KNNJaccard)


# In[ ]:





# In[ ]:





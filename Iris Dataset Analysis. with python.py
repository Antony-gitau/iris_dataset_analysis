#!/usr/bin/env python
# coding: utf-8

# This is my first ML classification problem am doing. I am working on the famous iris dataset. i hope to do some beginer analysis on this data set and also train a model.

# In[1]:


#start by importing the basic modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#loading the dataset

iris  = pd.read_csv("D:\MyLib.Doc\Programming.Doc\Python.allstuffs\Python for DataAnalysis\iris.csv" )


# In[5]:


#the first five rows of the iris data

iris.head()


# In[6]:


#display the statistics of aboot the data

iris.describe()


# In[7]:


#more about the iris data
#observe no null values

iris.info()


# In[10]:


#no of samples in every species

iris['species'].value_counts()


# In[11]:


#another way, though not as approriate

iris['species'].value_counts


# In[13]:


#preprocess the dataset

#checking for null values,

iris.isnull().sum()


# In[21]:


#data analysis
#visualize the data in graphs

#plot a histogram

iris['sepal_length'].hist()


# In[22]:


#this is a normal distribution curve

iris['sepal_width'].hist()


# In[23]:


iris['petal_length'].hist()


# In[24]:


iris['petal_width'].hist()


# In[25]:


#plotting scatter plots 

colors = ['red','blue','yellow']
species = ['Iris-virginica','Iris-setosa','Iris-versicolor']


# In[46]:


#iterate through the three classes and scatter them on the sam plot

for i in range(3):
    val=iris[iris['species']==species[i]]
    plt.scatter(val['sepal_length'], val['sepal_width'], 
                c= colors[i], label=species[i])
    
plt.xlabel('sepal_length',size=14)
plt.ylabel("sepal_width",size=14)
plt.title('Sepals info. in iris dataset', size=14)
plt.legend()
    


# In[48]:


#scattering the petals

for j in range(3):
    values = iris[iris['species'] == species[j]]
    plt.scatter(values['petal_length'], values['petal_width'], 
               c=colors[j], label=species[j])
    
plt.xlabel('petal_length', size=14)
plt.ylabel('petal_width', size=14)
plt.title("petal info. in iris dataset", size=14)
plt.legend()


# In[49]:


# coorelation matrix

iris.corr()


# In[50]:


#representing the coorrelation on a figure

corr = iris.corr()
fig,ax = plt.subplots(figsize=(5,5))
sns.heatmap(corr, annot=True, ax=ax)


# In[55]:


#label encoder
#converting the labels into numeric form

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
iris['species'] =le.fit_transform(iris['species'])


# In[56]:


#species has been converted to numeric form

iris.head()


# ## training a machine learning algorithm
# 

# In[149]:


#model training
#start by splitting the data into training and testing datasets

from sklearn.model_selection import train_test_split
X=iris.drop(columns=['species'])#training matrix
y=iris['species']#target vector
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)


# In[150]:


#using logistic regression 
#its a classification model

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
model= LogisticRegression()


# In[ ]:





# In[151]:


model.fit(X_train, y_train)


# In[152]:




model.score(X_test, y_test)


# In[153]:


#training KNeigbors classifier

from sklearn.neighbors import KNeighborsClassifier
kn_model=KNeighborsClassifier()


# In[154]:


kn_model.fit(X_train, y_train)


# In[155]:


kn_model.score(X_test, y_test)


# In[156]:


#training on decision tree classifier

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()


# In[157]:


dt_model.fit(X_train, y_train)


# In[158]:


dt_model.score(X_test,y_test)


# In[ ]:





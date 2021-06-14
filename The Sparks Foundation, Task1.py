#!/usr/bin/env python
# coding: utf-8

# ## NAME: Ankan Majumdar
#  Task 1 : Prediction using supervised learning.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[4]:


data


# In[5]:


data.describe()


# In[7]:


2.7 - 1.5*(7.4 - 2.7)


# In[8]:


7.4 + 1.5*(7.4 - 2.7)


# In[9]:


plt.scatter(data['Hours'],data['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[10]:


# To fit a straight thorugh given data such that the mean distance of all the observations from fitted lines is to be min.
# Eqution of Straight line is  m*x + c
# Cost function here becomes(m,c) = (sum i=1, n)(m*x + c - y)**2
# wrt to c=   (sum i = 1, n)2(m*x + c - y) = 0,       sum_x , sum_c, sum_y
# wrt to m=   (sum i = 1, n)2(m*x + c - y)*x = 0      sum_x^2 ,c*sum_x, sum_xy 


# In[11]:


data1 = data.rename(columns={'Hours':'Hours (x)','Scores':'Scores (y)'})


# In[13]:


data1.head()


# In[14]:


data1['x_square'] = np.array(data['Hours']**2)


# In[16]:


data1['xy'] = np.array(data['Hours']*data['Scores'])


# In[17]:


data1.head()


# In[18]:


m = len(data['Hours'])
sum_x = np.sum(data1['Hours (x)'])
sum_x_sq = np.sum(data1['x_square'])
sum_y = np.sum(data1['Scores (y)'])
sum_xy = np.sum(data1['xy'])


# In[19]:


# X*b = Y
# b = X^(-1)*y

X = np.array([[m , sum_x],[sum_x,sum_x_sq]])


# In[20]:


X


# In[21]:


Y = np.array([[sum_y],[sum_xy]])


# In[22]:


Y


# In[23]:


b = np.dot(np.linalg.inv(X),Y)


# In[24]:


b


# In[25]:


m = b[1]
c = b[0]
print(m)
print(c)


# In[29]:


hours = np.array(data['Hours'])
scores = np.array(data['Scores'])
line = m*hours + c
plt.scatter(hours,scores)
plt.plot(hours,line)
plt.show()


# In[30]:


predicted = []
for i in data['Hours']:
    j = i*m + c
    predicted.append(j)


# In[31]:


import math


# In[32]:


from sklearn.metrics import mean_squared_error


# In[33]:


rmse = math.sqrt(mean_squared_error(predicted,data['Scores']))
print(rmse)


# In[34]:


new = pd.DataFrame(data['Scores'])


# In[35]:


new['predicted scores'] = predicted


# In[36]:


new


# In[37]:


# Predict the score if study hours is 9.25hrs/day


# In[38]:


score = 9.25*m + c
print(score)


# In[ ]:


# If the student studies 9.25 hrs/day then he will 92.9% marks.


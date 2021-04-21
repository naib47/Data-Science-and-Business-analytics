#!/usr/bin/env python
# coding: utf-8

# <h3>GRIP: Spark Foundation</h3>
# <h2>Data science and Business Analytics interns in Apr_2021</h2>
# <h3>Author: Naib khan  </h3>
# <h4>Youtube channel link</h4> <link><ul>https://www.youtube.com/channel/UCDL4ZK5P4VyNLwgGN6-LmeA</ul></link>
# <h4>linked link link</h4> <link>https://www.linkedin.com/in/naib-khan-314545188/</link>
# <h3>Prediction using supervised Ml</h3>
# <p>in this task we have to predict to percantage score of the  student based on theof number of hours studied. This task contain's two variable where the features number of hurse studied  and target value is the perentage score. so by using simple regression </p>

# <h3>1. import libraries<h3>

# In[7]:


#import rewuired libraries fir regresion test using supervised Ml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <h3>2 read data set from local Machine</h3>

# In[8]:


path='Documents/IntersnShip SPARK FOundation/1 task prd using Seup Ml student/data set/student_scores - student_scores.csv'


# In[10]:


data=pd.read_csv(path)
print("The data imported from local Machine successfuly")


# In[11]:


#exploring data
print(data.shape)


# In[12]:


#head of data set
data.head()


# In[13]:


#getting data information by using following function
data.info()


# In[14]:


data.describe()


# In[15]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[16]:


data.corr(method='pearson')


# In[17]:


data.corr(method='spearman')


# In[18]:


Hours=data['Hours']
Scores=data['Scores']


# In[19]:


sns.distplot(Hours)


# In[20]:


sns.distplot(Scores)


# <p>Form the above Graph We clearly see that there is a clearly postive linear co relation between the student concerned data that student number of percntage increase as time scheule increase</p>
# <h3> 3 Preparing data by using linear r </h3>
#     

# <p>The next step is to divide the data into "attributes" (inputs) and "labels" (outputs)</p>

# In[21]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# <h3>4 Training Data </h3>

# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# <h3>5. training algorithm</h3>

# In[23]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
print('training algo successfuly')


# In[ ]:


#ploting the regression line


# In[24]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# <h3>6.Making prediction</h3>

# In[25]:



y_pred=reg.predict(X_test)
print(X_test)


# In[26]:


actual_predicted=pd.DataFrame({'actual':y_test,'Predicted':y_pred})
actual_predicted


# In[27]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# <p>what would be the predicted scores if a student studies for 9.25/daay</p>

# In[28]:


h=9.25
s=reg.predict([[9]])
print("if a student studies for a {} hours/day he/she will score {} % in exam" .format(h,s))


# <h3>7.Model Evaluation</h3>

# In[29]:


from sklearn import metrics  
from sklearn.metrics import r2_score
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
print('R2_score:',r2_score(y_test,y_pred))


# In[1]:


import os
print(os.environ['PATH'])


# In[ ]:





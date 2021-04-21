#!/usr/bin/env python
# coding: utf-8

# <h3>GRIP: Spark Foundation</h3>
# <h3>Data science and Business Analytics interns in Apr_2021</h3>
# <h3>Author: Naib khan </h3>
# <h3>Taks 3:: prediction using Un Supervised Machine learning</h3>
# <h4>Exploratory Data Analysis _Retail</h4>
# <p>Perform ‘Exploratory Data Analysis’ on dataset ‘SampleSuperstore’ </p>
# <p>● As a business manager, try to find out the weak areas where you can 
# work to make more profit. </p>
# <p>● What all business problems you can derive by exploring the data?</p>
# channnel link :   https://www.youtube.com/watch?v=9yX2ZdTZWZ0&t=64s

# <h3>1. importing libraries</h3>

# In[1]:


#video link https://www.youtube.com/watch?v=9yX2ZdTZWZ0&t=64s
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>2 : loading dataset</h3>
# 
# 

# In[2]:


df=pd.read_csv('SampleSuperstore.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()#cheacking Null values


# In[8]:


df.info()#information about data set


# In[8]:


df.columns


# In[9]:


df.duplicated().sum()


# In[10]:


df.nunique()


# In[11]:


df['Postal Code']=df['Postal Code'].astype('object')


# In[12]:


df.drop_duplicates(subset=None,keep='first',inplace=True)
df.duplicated().sum()


# In[13]:


corr=df.corr()


# In[14]:


sns.heatmap(corr,annot=True)


# In[15]:


df.columns


# In[17]:


sns.pairplot(df,hue='Ship Mode')


# In[16]:


df['Ship Mode'].value_counts()


# In[18]:


sns.countplot(x=df['Ship Mode'])


# In[19]:


sns.countplot(x=df['Ship Mode'])


# In[20]:


df['Segment'].value_counts()#values count for segment


# In[21]:


sns.pairplot(df,hue='Segment')#polting pair plot


# In[22]:


sns.countplot(x="Segment",data=df,palette='rainbow')


# In[23]:


df['Category'].value_counts()


# In[24]:


sns.countplot(x='Category',data=df,palette='tab10')


# In[27]:


sns.pairplot(df,hue='Category')


# In[25]:


df['Sub-Category'].value_counts()


# In[26]:


plt.figure(figsize=(15,12))


# In[27]:


df['Sub-Category'].value_counts().plot.pie(autopct='**')


# <h3>observation</h3>
# <p>Maximum are from Binders, paper ,Furnsihing,phone storage ,art,accessor</p>
# <p>Minimum from copier supplier etc till end </p>
# 

# In[31]:


df['State'].value_counts()


# In[32]:


plt.figure(figsize=(30,30))


# In[33]:


sns.countplot(x='State',data=df,palette='rocket_r',order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# <h3>Observation # 2</h3>
# <p>Higest Number of Buyers from California and Newyork</p>

# In[28]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# <h3>Observation # 3</h3>
# <p>-->Most custmoers tends to buy quantity of 2 and 3</p>
# <p>-->Maximum Dscount is given in between 0-20 percent

# In[29]:


plt.figure(figsize=(10,8))
df['Region'].value_counts().plot.pie(autopct='%1.1f%%' )
plt.show()


# <h3>Profit Vs Discount</h3>

# In[30]:


fig,ax=plt.subplots(figsize=(20,8))

ax.scatter(df['Sales'],df['Profit'])
ax.set_xlabel('Sales')
ax.set_ylabel('Profit')
plt.show()


# In[37]:


sns.lineplot(x='Discount',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# <h3>Observation #4</h3>

# --> No corealtion Between Profit and Discount

# <h3>Profit vs Discount</h3>

# In[38]:


sns.lineplot(x='Quantity',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# In[31]:


df.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['Pink','green'],figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# <h3>observation #5</h3>
# <p>-->Profit and sales are Maximum in Consumer segment</p>
# -->Minimum in Home Office Segment

# In[40]:


plt.figure(figsize=(12,8))
plt.title('Segment wise sales in each Region')
sns.barplot(x='Region',y='Sales',data=df, hue='Segment',order=df['Region'].value_counts().index,palette='rocket')
plt.xlabel('Region',fontsize=15)
plt.show()


# <h3>Observation 6</h3>
# <p>Segment wise salesa re almost same in every region</p>
# 

# In[41]:


df.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['blue','red'],figsize=(8,5))
plt.ylabel('Profit/loss and sales')
plt.show()


# <h3>Observation No:7</h3>
# <p>Profit and Sales are MAximum in west region and Miminum in South Region</p>

# In[32]:


ps=df.groupby('State')[['Sales','Profit']].sum().sort_values(by='Sales' ,ascending =False)


# In[43]:



ps[:].plot.bar(color=['blue','orange'], figsize=(30,16))
plt.title('Profit/loss and sales across states')
plt.xlabel('states')
plt.ylabel('Profit/loss and sales')
plt.show()


# <h3>Observation No.8</h3>
# <p>-->high profit is for California and Newyork</p>
# ==>Loss is for Texas,Pennsylvania and Ohio

# In[44]:


t_state=df['State'].value_counts().nlargest(10)
t_state


# In[45]:


df.groupby('Category')[['Profit','Sales']].sum().plot.bar(color=['yellow','purple'],alpha=0.9,figsize=(8,5) )


# <h3>Observation No:9</h3>
#     <p>Technology and office Supplies have high profi</p>
#     Furniture have less profit

# In[46]:


ps=df.groupby('Sub-Category')[['Sales','Profit']].sum().sort_values(by='Sales' ,ascending =False)
ps[:].plot.bar(color=['red','lightblue'],figsize=(15,8))
plt.title('Profit/loss and sales across state')
plt.xlabel('Sub-Category')
plt.ylabel('Profit/loss & Sales')
plt.show()


# <h3>observation No.10</h3>
# <p>Phones sub-category have high sales</p>
# <p>Table and Book Marks Sub-Category facing huge loss</p>
# <p>chairs have high sales but less profit compared to phones</p>
# <h2>Assignment work completed</h2>

# In[ ]:





# In[ ]:





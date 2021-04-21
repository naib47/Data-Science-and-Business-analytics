#!/usr/bin/env python
# coding: utf-8

# <h3>GRIP: Spark Foundation</h3>
# <h3>Data science and Business Analytics interns in Apr_2021</h3>
# <h3>Author: Naib khan </h3>
# 
# <h3>Taks 2:: prediction using Un Supervised Machine learning</h3>

# <p>in this task its required to predict the optimum number of cluster for the iris dataset  consist of three type of features iris_setosa,irs_versicolor,iris_virginicaa</p>
# 
# <h4>Steps</h4>
# <p>step 1: importing the dataset form local Machine</p>
# <p>step 2: visualization the dataset</p>
# <p>step 3: finding the optimum Number of cluster</p>
# <p>step 4: applying the K_means clustering on the data set</p>
# <p>step 5: visualizing the final optimum cluster</p>
# <h3>step 1: importing the dataset form local Machine</h3>

# In[106]:


#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sas
get_ipython().run_line_magic('matplotlib', 'inline')


# In[108]:


#load the iris data set
df=pd.read_csv("Iris.csv")


# In[109]:


df.head()


# In[76]:


df.tail()


# In[110]:


df.shape


# In[111]:


df.columns


# In[79]:


df['Species'].unique()


# In[112]:


df.info()


# <h3>step 2: visualization the dataset</h3>

# In[ ]:





# In[114]:


fig = plt.figure(figsize=(8,5))

plt.scatter(x = iris.Id, y = iris.SepalLengthCm, marker = "*")
plt.scatter(x = iris.Id, y = iris.SepalWidthCm, marker = "v")
plt.scatter(x = iris.Id, y = iris.PetalLengthCm, marker = "D")
plt.scatter(x = iris.Id, y = iris.PetalWidthCm, marker = "o")


fig.legend(labels=['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])


# 
# What is K ?
# K is the number of clusters. Here we will understand the meaning of K by classification of Sepal Length & width in 2 and 5 clusters and will measure the centroids of each cluster and plot them onto the graph.

# In[115]:


from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(iris.iloc[:, [0, 1]])


# In[116]:



Kmean.cluster_centers_


# In[117]:



Kmean.labels_


# In[85]:


Kmean.inertia_


# In[118]:


plt.scatter(iris.Id, iris.SepalLengthCm, s =50, c='y')
plt.scatter(113., 6.34533333,s = 200, c='b', marker='s')
plt.scatter(38, 5.34133333, s = 200, c='r', marker='s')
plt.show()


# In[87]:


from sklearn.cluster import KMeans
Kmean2 = KMeans(n_clusters=2)
Kmean2.fit(iris.iloc[:, [0, 2]])
Kmean2.cluster_centers_


# In[119]:


plt.scatter(iris.Id, iris.SepalWidthCm, s =50, c='y')
plt.scatter(113., 2.904,s = 200, c='b', marker='s')
plt.scatter(38, 3.204, s = 200, c='r', marker='s')
plt.show()


# In[89]:


from sklearn.cluster import KMeans
Kmean2 = KMeans(n_clusters=6)
Kmean2.fit(iris.iloc[:, [0, 1]])
Kmean2.cluster_centers_


# In[120]:


plt.scatter(iris.Id, iris.SepalLengthCm, s =50, c='y')
for i in range(0, 6):
    plt.scatter(Kmean2.cluster_centers_[i, 0], Kmean2.cluster_centers_[i, 1],s = 200, c='b', marker='s')
plt.show()


# <h4>How to determine the value of K ?</h4>
# <h4>Step 4 -> Finding the optimal number of clusters using elbow method</h4>
# The “elbow” method helps data scientists select the optimal number of clusters by fitting the model with a range of values for K. If the line chart resembles an arm, then the “elbow” (the point of inflection on the curve) is a good indication that the underlying model fits best at that point.
# 
# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variances, minimizing a criterion known as the inertia or within-cluster sum-of-squares Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are.
# 
# The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean j of the samples in the cluster. The means are commonly called the cluster centroids.
# 
# The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum of squared criterion:

# In[91]:


x = iris.iloc[:, [1, 2, 3, 4]].values

SumOfSquaresInertia = []

for i in range(1, 11):
    model = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    model.fit(x)
    SumOfSquaresInertia.append(model.inertia_)
    
plt.plot(range(1, 11), SumOfSquaresInertia)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WithinSumOfSquaresInertia') # Within cluster sum of squares
plt.show()


# <h3>observation</h3>You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as '3'.
# 
# <h3>Step 5 -> Creating the K-Means classifier</h3>

# In[122]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)


# In[123]:


y_kmeans


# <h3>Sepal Length Cm</h3>

# In[124]:


plt.figure(figsize=(8, 5))

plt.scatter(iris[y_kmeans == 0].Id, x[y_kmeans == 0, 0], 
            s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(iris[y_kmeans == 1].Id, x[y_kmeans == 1, 0], 
            s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(iris[y_kmeans == 2].Id, x[y_kmeans == 2, 0],
            s = 50, c = 'green', label = 'Iris-virginica')


# <h3>Step 5.1 -> Classifying for Speal Length and Sepal Width</h3>

# In[125]:


kmeans.cluster_centers_


# In[126]:


plt.figure(figsize=(6, 6))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 50, c = 'green', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 50, c = 'red', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')


# 
# <h3>Step 5.2 -> Classifying for other six combinations</h3>
# <p>1) Sepal Length vs. Sepal Width</p>
# <p>2) Sepal Length vs. Petal Length</p>
# <p>3) Sepal Length vs. Petal Width</p>
# <p>4) Sepal Width vs. Petal Length</p>
# <p>5) Sepal Width vs. Petal Width</p>
# <p>6) Petal Length vs. Petal Width</p>

# In[127]:



fig, axs = plt.subplots(2, 3, figsize = (16, 8))

m = 0
n = 0
for i in range(0, 4):
    for j in range(i+1, 4):
        if m < 3:
            axs[0, m].scatter(x[y_kmeans == 0, i],x[y_kmeans == 0, j], s = 50, c = 'green', label = 'Iris-setosa')
            axs[0, m].scatter(x[y_kmeans == 1, i], x[y_kmeans == 1, j],s = 50, c = 'blue', label = 'Iris-versicolour')
            axs[0, m].scatter(x[y_kmeans == 2, i], x[y_kmeans == 2, j],s = 50, c = 'red', label = 'Iris-verginica')
            axs[0, m].scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:,j], s = 100, c = 'yellow', label = 'Centroids', marker = "D")
            m = m + 1
        elif n < 3:
            axs[1, n].scatter(x[y_kmeans == 0, i],x[y_kmeans == 0, j], s = 50, c = 'green', label = 'Iris-setosa')
            axs[1, n].scatter(x[y_kmeans == 1, i], x[y_kmeans == 1, j],s = 50, c = 'blue', label = 'Iris-versicolour')
            axs[1, n].scatter(x[y_kmeans == 2, i], x[y_kmeans == 2, j],s = 50, c = 'red', label = 'Iris-verginica')
            axs[1, n].scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:,j], s = 100, c = 'yellow', label = 'Centroids', marker = "D")
            n = n + 1

axs[0, 0].set_title('Sepal Length vs. Sepal Width')
axs[0, 1].set_title('Sepal Length vs. Petal Length')
axs[0, 2].set_title('Sepal Length vs. Petal Width')
axs[1, 0].set_title('Sepal Width vs. Petal Length')
axs[1, 1].set_title('Sepal Width vs. Petal Width')
axs[1, 2].set_title('Petal Length vs. Petal Width')


# <h3>Step 6 -> Checking the accuracy of our prediction with origional classification</h3>

# In[128]:


iris2 = pd.read_csv("Iris.csv", usecols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])


# In[129]:


iris2.head(2)


# In[101]:


sas.pairplot(iris2, hue = "Species", corner = True)


# In[130]:


a = np.array([])

for i in range(0, 150):
    if iris2.Species[i] == 'Iris-setosa':
        a = np.append(a, [1])
    elif iris2.Species[i] == 'Iris-versicolor':
        a = np.append(a, [2])
    elif iris2.Species[i] == 'Iris-virginica':
        a = np.append(a, [0])


# In[131]:


y_kmeans


# In[104]:


a


# In[132]:


from sklearn.metrics import accuracy_score
print(accuracy_score(a, y_kmeans))
print(accuracy_score(a, y_kmeans, normalize = False))


# In[ ]:





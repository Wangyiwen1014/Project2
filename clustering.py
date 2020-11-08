#Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


#data loading
train=pd.read_csv('train(Response1).csv')
train = train.iloc[1:]


###Clustering for 'Age'&'Region_Code'

feature_cols1=['Age','Region_Code']

train1 = train[feature_cols1].values # Features

inertia1 = []# Store the sum of the squared errors of each clustering result

for n in range(1, 11):
    # Cluster model 1 Constructing
    km1 = (KMeans(n_clusters=n,        # Cluster number，int，default8
                  init='k-means++',    # Initializing the center of mass
                  n_init=10,           # Sets the number of times to select centroid seeds. Default 10 .Returns the best possible result for the center of mass.
                  max_iter=300,        # Maximum number of iterations
                  tol=0.1,             # The minimum tolerance error, when the error is less than TOL, iteration exits
                  random_state=111,   
                  algorithm='elkan'))  # Apply 'elkan'K-Means algorithm
    
    km1.fit(train1)

    inertia1.append(km1.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia1, 'o')
plt.plot(np.arange(1, 11), inertia1, '-', alpha=0.7)
plt.xlabel('Cluster number'), plt.ylabel('SSE')
plt.grid(linestyle='-.')
plt.show() #Show the elbow method diagram to determine the k value

km1_result = (KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300,
                      tol=0.1,  random_state=111, algorithm='elkan'))

y1_means = km1_result.fit_predict(train1)

plt.scatter(train1[y1_means == 0][:, 0], train1[y1_means == 0][:, 1], s=5, c='blue', label='1', alpha=0.6)
plt.scatter(train1[y1_means == 1][:, 0], train1[y1_means == 1][:, 1], s=5, c='orange', label='2', alpha=0.6)
plt.scatter(train1[y1_means == 2][:, 0], train1[y1_means == 2][:, 1], s=5, c='pink', label='3', alpha=0.6)
plt.scatter(train1[y1_means == 3][:, 0], train1[y1_means == 3][:, 1], s=5, c='purple', label='4', alpha=0.6)
plt.scatter(train1[y1_means == 4][:, 0], train1[y1_means == 4][:, 1], s=5, c='green', label='5', alpha=0.6)
plt.scatter(train1[y1_means == 5][:, 0], train1[y1_means == 5][:, 1], s=5, c='black', label='6', alpha=0.6)
plt.scatter(km1_result.cluster_centers_[:, 0], km1_result.cluster_centers_[:, 1], s=260, c='gold', label='Center')
plt.title('Clustering Map(K=6)', fontsize=12)
plt.xlabel('Age')
plt.ylabel('Region_Code')
plt.legend()
plt.grid(linestyle='-.')
plt.show()  #Show the Clustering map


###Clustering for 'Age'&''Policy_Sales_Channel''

feature_cols2=['Age','Policy_Sales_Channel']

train2 = train[feature_cols2].values # Features

inertia2 = [] # Store the sum of the squared errors of each clustering result

for n in range(1, 11):
    # Cluster model 2 Constructing
    km2 = (KMeans(n_clusters=n,        # Cluster number，int，default8
                  init='k-means++',    # Initializing the center of mass
                  n_init=10,           # Sets the number of times to select centroid seeds. Default 10 .Returns the best possible result for the center of mass.
                  max_iter=300,        # Maximum number of iterations
                  tol=0.1,             # The minimum tolerance error, when the error is less than TOL, iteration exits
                  random_state=111,    # 随机生成器的种子 ，和初始化中心有关
                  algorithm='elkan'))  # Apply 'elkan'K-Means algorithm
    
    km2.fit(train2)

    inertia2.append(km2.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia2, 'o')
plt.plot(np.arange(1, 11), inertia2, '-', alpha=0.7)
plt.xlabel('Cluster number'), plt.ylabel('SSE')
plt.grid(linestyle='-.')
plt.show() #Show the elbow method diagram to determine the k value

km2_result = (KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                      tol=0.1,  random_state=111, algorithm='elkan'))

y2_means = km2_result.fit_predict(train2)

plt.scatter(train2[y2_means == 0][:, 0], train2[y2_means == 0][:, 1], s=5, c='blue', label='1', alpha=0.6)
plt.scatter(train2[y2_means == 1][:, 0], train2[y2_means == 1][:, 1], s=5, c='orange', label='2', alpha=0.6)
plt.scatter(train2[y2_means == 2][:, 0], train2[y2_means == 2][:, 1], s=5, c='pink', label='3', alpha=0.6)
plt.scatter(km2_result.cluster_centers_[:, 0], km2_result.cluster_centers_[:, 1], s=260, c='gold', label='Center')
plt.title('Clustering Map(K=3)', fontsize=12)
plt.xlabel('Age')
plt.ylabel('Policy_Sales_Channel')
plt.legend()
plt.grid(linestyle='-.')
plt.show()  #Show the Clustering map

###Clustering for 'Region_code'&'Policy_Sales_Channel'


feature_cols3=['Region_Code','Policy_Sales_Channel']

train3 = train[feature_cols3].values # Features

inertia3 = [] # Store the sum of the squared errors of each clustering result

for n in range(1, 11):
    # Cluster model 3 Constructing
    km3 = (KMeans(n_clusters=n,        # Cluster number，int，default8
                  init='k-means++',    # Initializing the center of mass
                  n_init=10,           # Sets the number of times to select centroid seeds. Default 10 .Returns the best possible result for the center of mass.
                  max_iter=300,        # Maximum number of iterations
                  tol=0.1,             # The minimum tolerance error, when the error is less than TOL, iteration exits
                  random_state=111,    # 随机生成器的种子 ，和初始化中心有关
                  algorithm='elkan'))  # Apply 'elkan'K-Means algorithm
    
    km3.fit(train3)

    inertia3.append(km3.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia3, 'o')
plt.plot(np.arange(1, 11), inertia3, '-', alpha=0.7)
plt.xlabel('Cluster number'), plt.ylabel('SSE')
plt.grid(linestyle='-.')
plt.show() #Show the elbow method diagram to determine the k value

km3_result = (KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                      tol=0.1,  random_state=111, algorithm='elkan'))

y3_means = km3_result.fit_predict(train3)

plt.scatter(train3[y3_means == 0][:, 0], train3[y3_means == 0][:, 1], s=5, c='blue', label='1', alpha=0.6)
plt.scatter(train3[y3_means == 1][:, 0], train3[y3_means == 1][:, 1], s=5, c='orange', label='2', alpha=0.6)
plt.scatter(train3[y3_means == 2][:, 0], train3[y3_means == 2][:, 1], s=5, c='pink', label='3', alpha=0.6)
plt.scatter(km3_result.cluster_centers_[:, 0], km3_result.cluster_centers_[:, 1], s=50, c='gold', label='Center')
plt.title('Clustering Map(K=3)', fontsize=12)
plt.xlabel('Region_Code')
plt.ylabel('Policy_Sales_Channel')
plt.legend()
plt.grid(linestyle='-.')
plt.show()  #Show the Clustering map

### isualizing'Agw'&'Region_code'&'Policy_Sales_Channel'
feature_cols4=['Age','Region_Code','Policy_Sales_Channel']
train4 = train[feature_cols4].values # Features
km4 = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
km4.fit(train4)
train['labels'] = km4.labels_
# 绘制3D图
trace1 = go.Scatter3d(
    x=train['Age'],
    y=train['Region_Code'],
    z=train['Policy_Sales_Channel'],
    mode='markers',
      marker=dict(
        color=train['labels'],
        size=10,
        line=dict(
            color=train['labels'],
            width=12
        ),
        opacity=0.8
      )
)
df_3dfid = [trace1]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene=dict(
            xaxis=dict(title='Age'),
            yaxis=dict(title='Region_Code'),
            zaxis=dict(title='Policy_Sales_Channel')
        )
)

fig = go.Figure(data=df_3dfid, layout=layout)
py.offline.plot(fig)


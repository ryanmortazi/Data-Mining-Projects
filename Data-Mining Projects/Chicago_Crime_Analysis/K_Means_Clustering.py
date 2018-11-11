# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:39:04 2018

@author: RyanM
"""

# =============================================================================
#SOURCE: http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
# Use pd.concat() to join the columns and then drop() 
# the original country column:

# import pandas as pd
# # df now has two columns: name and country
# df = pd.DataFrame({
#         'name': ['josef','michael','john','bawool','klaus'],
#         'country': ['russia', 'germany', 'australia','korea','germany']
#     })
# 
# # use pd.concat to join the new columns with your original dataframe
# df = pd.concat([df,pd.get_dummies(df['country'], prefix='country')],axis=1)
# 
# # now drop the original 'country' column (you don't need it anymore)
# df.drop(['country'],axis=1, inplace=True)
# =============================================================================

# =============================================================================
#SOURCE: http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
# Treat Nulls/NaNs as a separate category
# import pandas as pd
# import numpy as np
# 
# df = pd.DataFrame({
#     'country': ['germany',np.nan,'germany','united kingdom','america','united kingdom']
# })
# 
# pd.get_dummies(df,dummy_na=True)
# =============================================================================

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data=pd.read_csv("Chicago_Crimes_2001_to_2004.csv",  error_bad_lines=False)
data.head()
dataset=data.iloc[:,[5,14]].copy()
dataset.head()

# =============================================================================
# preprocessing data
# =============================================================================
dataset["IUCR"].value_counts()
dataset["Community Area"].value_counts()
cleanup_nums = {"IUCR":     {"031A": 313, "031B": 313,"033A": 337, "033B": 337,
                             "041A": 420, "041B": 420,
                             "051A": 520, "051B": 520, "141A": 1435, "141B": 1435,
                             "141C": 1435, "142A": 1435, "142B": 1435,
                             "143A": 1435, "143B": 1435, "143C": 1435,
                             "500E": 5011,"500N": 5011,"501A": 5011,"501H": 5011,
                             "502P": 5011,"502R": 5011,"502T": 5011}}
dataset.replace(cleanup_nums, inplace=True)
#dataset=dataset.fillna({"Community Area": 0})
dataset=dataset.dropna()
dataset=dataset[dataset["Community Area"]>0]
dataset["IUCR"] = dataset["IUCR"].astype(str)
df=dataset.copy()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
dataset.iloc[:,0]=le.fit_transform(dataset.iloc[:,0])
dataset.iloc[:,1]=le.fit_transform(dataset.iloc[:,1])
#dataset.iloc[:,2]=le.fit_transform(dataset.iloc[:,2])
#oneHotEncoder=OneHotEncoder(categorical_features=[1])
#dataset=oneHotEncoder.fit_transform(dataset).toarray()
# =============================================================================
# Sclaing data
# =============================================================================
scaledDataset=scale(dataset)

# =============================================================================
#  Visualize the results on PCA-reduced data
# =============================================================================

reduced_data = PCA(n_components=2).fit_transform(scaledDataset)

# =============================================================================
# Using the elbow method to find the optimal number of clusters
# =============================================================================
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# =============================================================================
#  Fitting K-Means to the dataset (Udemy Style)
# =============================================================================
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset)

X=dataset.iloc[:,[0,1]].values
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 1], X[y_kmeans == 0, 0], s = 100, c = 'black', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1, 0], s = 100, c = 'blue',label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 1], X[y_kmeans == 2, 0], s = 100, c = 'green',label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 1], X[y_kmeans == 3, 0], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 1], X[y_kmeans == 4, 0], s = 100, c = 'magenta',label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 1], X[y_kmeans == 5, 0], s = 100, c = 'brown', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')
plt.title('K-means clustering on the Chicago Crime \n Centroids are marked with white cross')
plt.ylabel('IUCR (crime class)')
plt.xlabel('Community Area 1-77)')
plt.legend()
plt.show()

### to DO: analyse max and min of each cluster for community area and IUCR

for i in range(0,6):
    print("For the cluster # {}, IUCR is between {} and {}".
          format(i+1,df.iloc[y_kmeans == i,0].min(),df.iloc[y_kmeans == i,0].max()))
    print("For the cluster # {}, Community Area is between {} and {}".
          format(i+1,df.iloc[y_kmeans == i,1].min(),df.iloc[y_kmeans == i,1].max()))
          
# =============================================================================
# using K-Means
# =============================================================================
# =============================================================================
# kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
# kmeans.fit(reduced_data)
# 
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
# 
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# 
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
# 
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the Chicago Crime scaled dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
# =============================================================================

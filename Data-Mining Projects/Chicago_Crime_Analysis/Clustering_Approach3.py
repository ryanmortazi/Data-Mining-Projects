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
dF=data.copy()
dataset=data.iloc[:,[5,14,18]].copy()
dataset.head()

# =============================================================================
# preprocessing data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
dataset.iloc[:,0]=le.fit_transform(dataset.iloc[:,0])
dataset=dataset.fillna({"Community Area": 0})
dataset.iloc[:,2]=le.fit_transform(dataset.iloc[:,2])
dataset.iloc[:,1]=le.fit_transform(dataset.iloc[:,1])
oneHotEncoder=OneHotEncoder(categorical_features=[1])
dataset=oneHotEncoder.fit_transform(dataset).toarray()
# =============================================================================
# Sclaing data
# =============================================================================
scaledDataset=scale(dataset)

# =============================================================================
#  Visualize the results on PCA-reduced data
# =============================================================================

reduced_data = PCA(n_components=2).fit_transform(scaledDataset)
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the Chicago Crime scaled dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
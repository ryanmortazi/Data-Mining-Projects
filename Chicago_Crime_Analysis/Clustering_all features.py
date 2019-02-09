# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:37:13 2018

@author: RyanM
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pandas as pd
from scipy.stats.mstats import mquantiles, kurtosis, skew
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset2004=pd.read_csv("Chicago_Crimes_2001_to_2004.csv",  error_bad_lines=False)
dataset2004_Mod= dataset2004.iloc[:,[6,8,9,10,14]].copy()
dataset2004_Mod.head()
dataset2004_Mod.dtypes
# =============================================================================
# obj_df = dataset2004.select_dtypes(include=['object']).copy()
# obj_df["Primary Type"].value_counts()
# =============================================================================
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("RESIDE"),"residence",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("SCHOOL"),"school/college/library",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("COLLEGE"),"school/college/library",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("LIBRARY"),"school/college/library",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("STORE"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("WAREHOUSE"),"factory",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("FACTORY"),"factory",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("AIRPORT"),"AIRPORT",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CTA"),"Public Transportation",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TRANSPORT"),"Public Transportation",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("OFFICE"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CHA"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("PARK PROPERTY"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GAS STATION"),"private property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("SIDEWALK"),"STREET",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CHURCH"),"CHURCH",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("ALLEY"),"STREET",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("MOTEL"),"HOTEL/MOTEL",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("HOTEL"),"HOTEL/MOTEL",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TAVERN"),"BAR OR TAVERN",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("LOT"),"VACANT LOT/BUILDING",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CONSTRUCTION"),"VACANT LOT/BUILDING",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("ABANDONED"),"VACANT LOT/BUILDING",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("VEHICLE"),"VEHICLE/TRUCK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GARAGE"),"GARAGE",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("BARBER"),"BARBERSHOP",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CLUB"),"CLUB",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("POOL ROOM"),"CLUB",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("AUTO"),"VEHICLE/TRUCK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CARE"),"HOSPITALS/CLINICS",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("HOSPITAL"),"HOSPITALS/CLINICS",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TAdataset2004_ModI CAB"),"VEHICLE/TRUCK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TRUCK"),"VEHICLE/TRUCK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CREDIT"),"BANK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("SAVING"),"BANK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("BASEMENT"),"HOUSE",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("PORCH"),"HOUSE",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("DRIVEWAY"),"HOUSE",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("YMCA"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("LAUNDRY ROOM"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("NEWSSTAND"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CARWASH"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("BARBERSHOP"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("PRAIRIE"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("LAKE"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GANG"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("BRIDGE"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TRAILER"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("FUNERAL"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("VESTIBUL"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("JAIL"),"JAIL",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("RIVER"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CEMETARY"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("DUMPSTER"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GARBAGE DUMP"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("SEWER"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("WOODED"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("LOADING"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("COACH HOUSE"),"private property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("HALLWAY"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("STAIRWELL"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("RAILROAD"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("HIGHWAY"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("YARD"),"OTHER",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("CAR WASH"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GARAGE"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("PAWN SHOP"),"PRIVATE BUSINESS PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("FEDERAL"),"FEDERAL/GOVERNMNET PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("GOVERNMENT"),"FEDERAL/GOVERNMNET PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("FOREST"),"public property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("BOAT"),"private property crime",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("TAdataset2004_ModI"),"VEHICLE/TRUCK",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("FIRE STATION"),"FEDERAL/GOVERNMNET PROPERTY",dataset2004_Mod["Location Description"])
dataset2004_Mod["Location Description"] = np.where(dataset2004_Mod["Location Description"].
           str.contains("COIN"),"OTHER",dataset2004_Mod["Location Description"])
locationDescription_counts=dataset2004_Mod["Location Description"].value_counts()

dataset2004_Mod=pd.get_dummies(dataset2004_Mod, columns=["Primary Type"], 
                               prefix=["crimeType"])
dataset2004_Mod=pd.get_dummies(dataset2004_Mod, columns=["Community Area"], 
                               prefix=["Community Area"])
dataset2004_Mod=pd.get_dummies(dataset2004_Mod, columns=["Location Description"], 
                               prefix=["Location"])
'''
K_Means
'''
from sklearn.cluster import KMeans
wcss = []
for i in range(4, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(dataset2004_Mod)
    wcss.append(kmeans.inertia_)
plt.plot(range(4, 11), wcss)
plt.title('The Elbow Method')
plt.dataset2004_Modlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset2004_Mod)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], dataset2004_Mod[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(dataset2004_Mod[y_kmeans == 1, 0], dataset2004_Mod[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(dataset2004_Mod[y_kmeans == 2, 0], dataset2004_Mod[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(dataset2004_Mod[y_kmeans == 3, 0], dataset2004_Mod[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(dataset2004_Mod[y_kmeans == 4, 0], dataset2004_Mod[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.dataset2004_Modlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

'''
Decision tree regression
'''
'''
Support vector regression (SVR)
'''
'''
Random Forest regression 
'''
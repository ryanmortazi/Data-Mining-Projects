# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:17:07 2018

@author: RyanM
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pandas as pd
from kmodes.kmodes import KModes

data=pd.read_csv("Chicago_Crimes_2001_to_2004.csv",  error_bad_lines=False)
dF=data.copy()
data=data.drop(["Unnamed: 0","ID","Case Number","Latitude","Longitude"],axis=1)

# =============================================================================
# subsetData= data[data['Year']==(2001)].values
# subsetData.head()
# =============================================================================

# =============================================================================
# def filterYear(year):
#     df= finalDf[finalDf.Year==year]
#     return df
# =============================================================================
def makeClusters (data,year, numClusters):
    km=KModes(n_clusters=numClusters, init="Cao", n_init=1, verbose=1)
    subsetData= data[data["Year"]==year].drop(["Year","Community Area","Beat"],axis=1).values
    fitClusters=km.fit_predict(subsetData)
    clustersCentroidsData=pd.DataFrame(km.cluster_centroids_)
    clustersCentroidsData.columns=subsetData.columns
    return fitClusters, clustersCentroidsData

def labelDf(year, num, originalData, clusterInfo):
    df=originalData[originalData["Year"]==year]
    df=df.reset_index()
    clusterDf=pd.DataFrame(clusterInfo)
    clusterDf.columns=["Clusters_"+ str(num)]
    combinedDf=pd.concat([df,clusterDf],axis=1).reset_index()
    combinedDf=combinedDf.drop(["index","level_O"],axis=1)
    return combinedDf

dataset_Prep= data.iloc[:,[6,7,8,9,10,11,13,17]].copy()

dataset_Prep["District"]=pd.DataFrame(dataset_Prep["District"]).astype(str)
dataset_Prep.dtypes
subsetData= dataset_Prep[dataset_Prep["Year"]==2001].drop(["Year","Community Area","Beat"],axis=1).values
km=KModes(n_clusters=5, init="Cao", n_init=1, verbose=1)
subsetData.dtypes
fitClusters=km.fit_predict(subsetData)
clustersData2001_5Clusters=makeClusters(dataset_Prep,2001,5)
    
    
# =============================================================================
# dataset2004_Mod=pd.get_dummies(dataset2004_Mod, columns=["Primary Type"], 
#                                prefix=["crimeType"])
# dataset2004_Mod=dataset2004_Mod.fillna({"Community Area": 0})
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(dataset2004_Mod)
# =============================================================================

# Using the elbow method to find the optimal number of clusters




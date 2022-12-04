#-------------------------------------------------------------------------
# AUTHOR: Abdur Rahman
# FILENAME: clustering.py
# SPECIFICATION: Take in data from a file and use the kmeans algorithm to group the data into clusters.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.iloc[:]

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

maxSilhouetteCoef = 0
kForMaxCoef = 2
allCoefs = []
bestKMeans = KMeans
for i in range(2,21):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit((X_training))

    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    # --> add your Python code here
    currentSilhouetteCoef = silhouette_score(X_training,kmeans.labels_)
    allCoefs.append((i,currentSilhouetteCoef))
    if currentSilhouetteCoef > maxSilhouetteCoef:
        maxSilhouetteCoef = currentSilhouetteCoef
        kForMaxCoef = i
        bestKMeans = kmeans



print(maxSilhouetteCoef)
print(kForMaxCoef)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.scatter(*zip(*allCoefs))
plt.ylabel("Silhouette Coefficient")
plt.xlabel("k value")
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
reshaped = np.array(df_test.values).reshape(1,len(df_test.index))[0]

#Calculate and print the Homogeneity of this kmeans clustering
#print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(reshaped, bestKMeans.labels_).__str__())

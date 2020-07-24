import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing data
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

######### Train data scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, 0:] = sc.fit_transform(X[:, 0:])

cov_mat = np.dot(X, X.T)
U,s,VT = np.linalg.svd(cov_mat)
k = 2
reduced_eig_vectors = U[:, 0:k]
#dimension reduction
reduced_mat = np.dot(cov_mat, reduced_eig_vectors)
print("Given data size:",X.shape,"\nReduced data size:", reduced_mat.shape)

#plotting the reduced 2d data
plt.plot(reduced_mat[:, 0], reduced_mat[:, 1], '*')
plt.title("14d data reduced into 2d")
plt.xlabel("1st principal direction of given data")
plt.plot("2nd principal direction of given data")
plt.show()

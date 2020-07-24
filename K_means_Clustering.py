import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from qpsolvers import solve_qp
from sklearn.metrics import confusion_matrix
from mpl_toolkits import mplot3d 
import random

#Importing Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:].values

k = [1, 2, 3, 4, 5, 6, 7, 8]
Cost_points = np.zeros(len(k))
for n in range(len(k)):
    min_cost = np.zeros(10)
    for p in range(10):
        mu = np.zeros((k[n], X.shape[1]))
        indices = random.sample(range(len(X)), k[n])
        Class = []
        for i in range(k[n]):
            Class.append([])
            mu[i, :] = X[indices[i]]
        No_iter = 30
        Cost_fun = np.zeros(No_iter)
        for m in range(No_iter):
            for j in range(len(X)):
                dis = np.zeros(k[n])
                for l in range(k[n]):
                    dis[l] = np.linalg.norm(X[j, :]- mu[l, :])
                Class[np.argmin(dis)].append(j)
            Cost = np.zeros(k[n])
            for i in range(k[n]):
                mu[i, :] = 1/len(X[Class[i]])*sum(X[Class[i]])
                Cost[i] = sum(sum(((X[Class[i], :]-mu[i, :])**2).T))
                Class[i] = []
            Cost_fun[m] = sum(Cost)
        min_cost[p] = Cost_fun[-1]
    Cost_points[n] = np.min(min_cost)
plt.plot(k, Cost_points)
plt.title("Cost vs different k(Elbow graph)")
plt.xlabel("k")
plt.ylabel("cost")
plt.show()

print("Elbow graph shows data can be clustered in 5 categories")

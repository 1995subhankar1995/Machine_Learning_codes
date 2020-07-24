import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

###########  Importing Dataset    #######################
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 1)
n = X_train.shape[1] + 1
for i in range(n):
    temp1 = X_train[:, i]**2
    temp2 = X_test[:, i]**2
    temp1 = temp1.reshape(len(X_train), 1)
    temp2 = temp2.reshape(len(X_test), 1)
    X_train = np.hstack((X_train, temp1))
    X_test = np.hstack((X_test, temp2))
X_train_ones = np.ones((len(X_train), 1))
X_test_ones = np.ones((len(X_test), 1))
X_train = np.hstack((X_train_ones, X_train))
X_test = np.hstack((X_test_ones, X_test))
#########  Only features scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc.transform(X_test[:, 1:])

def cost_fun(X, y, theta):
    return 1/(2*len(X))*(sum((np.dot(X, theta)-y)**2))
def grad_fun(X, y, theta):
    return (1/len(X))*np.dot(X.T,(np.dot(X, theta)-y))
def theta_update(X, y, theta, alpha):
    return theta - alpha*grad_fun(X, y, theta)

theta = np.zeros(X_train.shape[1])
Cost_List = []
alpha = 3*1e-3
Iter_List = []
for j in range(1500):
    temp = cost_fun(X_train, y_train, theta)
    if(temp <= 0):
        break
    else:
        Cost_List.append(temp)
        Iter_List.append(j)
        theta = theta_update(X_train, y_train, theta, alpha)
plt.scatter(X_train[:, 1], y_train, color = 'red')
plt.plot(X_train[:, 1], np.dot(X_train, theta), '*')
plt.title('Polynomial regression with degree 4 in train data')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
plt.plot(X_test[:, 1], X_test, '-')
plt.plot(X_train[:, 1], np.dot(X_train, theta), '*')
plt.title('Polynomial regression with degree 4 in test data')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
Cost = np.array(Cost_List)
Iter = np.array(Iter_List)
plt.plot(Iter, Cost)
plt.title('Cost vs No of iterations')
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.show()
error = (1/(2*len(X_test)))*sum((np.dot(X_test, theta)-y_test)**2)
##### predicting salary for 6.5th level
X_new = np.array([[20, 50, 1000, 60]])
for i in range(n):
    temp3 = X_new[0, i]**2
    temp3 = temp3.reshape(len(X_new), 1)
    X_new = np.hstack((X_new, temp3))
X_new[:, 0:] = sc.transform(X_new[:, 0:])
one = np.array([[1]])
X_new = np.hstack((one, X_new))
print("Salary is:\n", np.dot(X_new,theta))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)
######### Add the X0 column ################################
ones_train = np.ones((len(X_train),1))
ones_test = np.ones((len(X_test),1))

X_train = np.hstack((ones_train, X_train))
X_test = np.hstack((ones_test, X_test))


def cost_fun(X, y, theta):
    return 1/(2*len(X))*(sum((np.dot(X, theta)-y)**2))
def grad_fun(X, y, theta):
    return (1/len(X))*np.dot(X.T,(np.dot(X, theta)-y))
def theta_update(X, y, theta, alpha):
    return theta - alpha*grad_fun(X, y, theta)
theta = np.zeros(2)
Cost_List = []
alpha = 0.0001
Iter_List = []
for j in range(1000):
    temp = cost_fun(X_train, y_train, theta)
    if(temp <= 0):
        break
    else:
        Cost_List.append(temp)
        Iter_List.append(j)
        theta = theta_update(X_train, y_train, theta, alpha)
Cost = np.array(Cost_List)
Iter = np.array(Iter_List)
plt.plot(Iter, Cost)
plt.title('Cost vs No of iterations')
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.show()

###  Error in test data ##############
error = (1/(2*len(X_test)))*sum((np.dot(X_test, theta)-y_test)**2)
plt.scatter(X_train[:, 1], y_train, color = 'red')
plt.plot(X_train[:, 1], np.dot(X_train, theta), color = 'black')
plt.title('Salary vs Experience(linear fit on train data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
plt.plot(X_test[:, 1], y_test, 'o')
plt.plot(X_train[:, 1], np.dot(X_train, theta), '-')
plt.title('Salary vs Experience(linear fit on test data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

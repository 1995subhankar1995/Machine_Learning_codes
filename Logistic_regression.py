import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import confusion_matrix
###########  Importing Dataset    #######################
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
for i in range(len(X)):
    if(y[i] == 2):
        y[i] = 0
    else:
        y[i] = 1
########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)
######### Train data scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:] = sc.fit_transform(X_train[:, 0:])
X_test[:, 0:] = sc.transform(X_test[:, 0:])

ones_train = np.ones((len(X_train), 1))
ones_test = np.ones((len(X_test), 1))
X_train = np.hstack((ones_train,X_train))
X_test = np.hstack((ones_test,X_test))

def Sigmoid(z):
    return 1/(1 + np.exp(-z))
def Sigmoid_grad(Z):
    return Sigmoid(z)*(1-Sigmoid(z))
def cost_fun(X, y, theta):
    return (np.dot(y, np.log(Sigmoid(np.dot(X,theta))))
                    +np.dot((1-y), np.log(1-Sigmoid(np.dot(X,theta)))))
def Cost_grad(X, y, theta):
    return np.dot(X.T, (y-Sigmoid(np.dot(X,theta))))
def theta_update(X, y, theta, alpha):
    return theta + alpha* Cost_grad(X, y, theta)
def Accuracy(X, y, theta):
    prob_each_data = Sigmoid(np.dot(X, theta))
    y_pred = np.zeros(len(X))
    for i in range(len(X)):
        if(prob_each_data[i] >= 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    matrix = confusion_matrix(y, y_pred)
    return (1/len(X))*sum(y_pred == y)*100, matrix
theta = np.zeros(X_train.shape[1])
Cost_List = []
alpha = 1e-3
Iter_List = []
for j in range(200):
    temp = cost_fun(X_train, y_train, theta)
    if(temp >= 0):
        break
    else:
        Cost_List.append(temp)
        Iter_List.append(j)
        theta = theta_update(X_train, y_train, theta, alpha)
Cost = np.array(Cost_List)
Iter = np.array(Iter_List)
plt.plot(Iter, Cost)
print("Here Cost function is a concave")
plt.title('Cost vs No of iterations')
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.show()
Accuracy_train,Confusion_train = Accuracy(X_train, y_train, theta)
Accuracy_test,Confusion_test = Accuracy(X_test, y_test, theta)
print("Train accuracy:", Accuracy_train,"%\n","Test accuracy:",
      Accuracy_test, "%\n")
########### Predict in new data ##########
X_new = np.array([[1036172,2,1,1,1,2,1,2,1,1]])
X_new = sc.transform(X_new)
one = np.ones((1, 1))
X_new = np.hstack((one, X_new))
Prob_X_new = Sigmoid(np.dot(X_new, theta))
y_pred = (Prob_X_new >= 0.5)
print(y_pred)
print(Confusion_train,"\n", Confusion_test)


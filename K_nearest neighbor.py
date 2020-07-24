import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import confusion_matrix
###########  Importing Dataset    #######################
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)


def KNN(X_train, X_test, y_train, k):
    D = np.zeros(len(X_train))
    y_pred = np.zeros(len(X_test))
    std= np.zeros(X_train.shape[1])
    for i in range((X_train.shape[1])):
        std[i] = np.std(X_train[:, i])
    for i in range(len(X_train)):
        temp = 0
        for j in range(X_train.shape[1]):
            temp +=((X_train[i, j]-X_test[j])**2)/(std[j])**2
        D[i] = temp
    arg_indices = (np.argsort(D))[0:k]
    y_KNN = y_train[arg_indices]
    return int((sum(y_KNN == 1) > sum(y_KNN == 0)))
    

def Accuracy(X, y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    return (1/len(X))*sum(y_actual == y_pred)*100, matrix
k = 13
y_predict_train = np.zeros(len(X_train))
y_predict_test = np.zeros(len(X_test))
for i in range(len(X_train)):
    y_predict_train[i] = KNN(X_train, X_train[i,:], y_train, k)
for i in range(len(X_test)):
    y_predict_test[i] = KNN(X_train, X_test[i,:], y_train, k)

accuracy1, confuse1 = Accuracy(X_train, y_train, y_predict_train)
accuracy2, test_confuse = Accuracy(X_test, y_test, y_predict_test)
print("Train accuracy:", accuracy1,"%\n","Test accuracy:",
      accuracy2, "%\n")
print("Train confusion matrix:\n",confuse1, "\n",
      "Train confusion matrix:\n", test_confuse)

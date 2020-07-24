import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from qpsolvers import solve_qp
from sklearn.metrics import confusion_matrix

###########  Importing Dataset    #######################
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
for i in range(len(X)):
    if(y[i] == 0):
        y[i] = -1
########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)

######### Train data scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:] = sc.fit_transform(X_train[:, 0:])
X_test[:, 0:] = sc.transform(X_test[:, 0:])

def Kernel(X, y):
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = y[i]*np.dot(X[i, :], X[j, :])*y[j]
    return K
def SVM_learner(X, y, C):
    P = Kernel(X, y)
    p1 = 1e-5*np.identity(len(X))
    P = P + p1
    q = -1*np.ones(len(X))
    A = y;
    A = A.reshape(1, len(X))
    u = np.zeros(1)
    h1 = C*np.ones(len(X))
    h2 = np.zeros(len(X))
    h = np.concatenate((h1, h2))
    G1 = np.identity(len(X_train))
    G2 = -1*np.identity(len(X_train))
    G = np.concatenate((G1, G2))
    alphas = solve_qp(P, q, G, h, A, u)
    SV_indices = []
    for i in range(len(X)):
        if(alphas[i] >= 0.001):
            SV_indices.append(i)
    SV = X[SV_indices]
    SV_labels = y[SV_indices]
    SV_alphas = alphas[SV_indices]
    W = np.zeros(X.shape[1])
    for i in range(len(SV_alphas)):
        W += SV_alphas[i]*SV_labels[i]*SV[i]
    b = SV_labels[25] - np.dot(W, SV[25])
    class model_struct:
        pass
    model = model_struct()
    model.W = W
    model.b = b
    model.SV = SV
    model.SV_labels = SV_labels
    model.SV_alphas = SV_alphas
    return model
def Prediction(X, model):
    return np.sign(np.dot(X, model.W) + model.b)
def Accuracy(X, y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    return (1/len(X))*sum(y_actual == y_pred)*100, matrix
C = 1
model = SVM_learner(X_train, y_train, C)
y_predict_train = Prediction(X_train, model)
y_predict_test = Prediction(X_test, model)
accuracy1, confuse1 = Accuracy(X_train, y_train, y_predict_train)
accuracy2, test_confuse = Accuracy(X_test, y_test, y_predict_test)
print("Train accuracy:", accuracy1,"%\n","Test accuracy:",
      accuracy2, "%\n")
print("Train confusion matrix:\n",confuse1, "\n",
      "Test confusion matrix:\n", test_confuse)

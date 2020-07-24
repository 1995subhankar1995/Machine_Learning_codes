import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from qpsolvers import solve_qp
from sklearn.metrics import confusion_matrix

###########  Importing Dataset    #######################
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_0 = X[np.where(y == 0)]
X_1 = X[np.where(y == 1)]
########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)
######### Train data scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:] = sc.fit_transform(X_train[:, 0:])
X_test[:, 0:] = sc.transform(X_test[:, 0:])
X_train_0 = X_train[np.where(y_train == 0)]
X_train_1 = X_train[np.where(y_train == 1)]
y_train_0 = y_train[np.where(y_train == 0)]
y_train_1 = y_train[np.where(y_train == 1)]
######### Calculation of mean and standard deviation ######
mu_0 = (1/len(X_train_0))*sum(X_train_0)
mu_1 = (1/len(X_train_1))*sum(X_train_1)

prior_0 = len(X_train_0)/len(X_train)
prior_1 = len(X_train_1)/len(X_train)

m_0 = np.zeros(X_train_0.shape[1])
std_0 = np.zeros(X_train_0.shape[1])
for i in range(X_train_0.shape[1]):
    m_0[i] = np.mean(X_train_0[:, i])
    std_0[i] = np.std(X_train_0[:, i])
m_1 = np.zeros(X_train_1.shape[1])
std_1 = np.zeros(X_train_1.shape[1])
for i in range(X_train_1.shape[1]):
    m_1[i] = np.mean(X_train_1[:, i])
    std_1[i] = np.std(X_train_1[:, i])

class struct_model:
    pass
model = struct_model()
model.mu_0 = mu_0
model.mu_1 = mu_1
model.prior_0 = prior_0
model.prior_1 = prior_1
model.m_0 = m_0
model.m_1 = m_1
model.std_0 = std_0
model.std_1 = std_1

def gaussian(x, mu, std):
    return 1/np.sqrt(2*np.pi*std)*(np.exp(-(0.5*(x-mu)**2)/std**2))

def Predictions(X, model):
    y_pred = np.zeros(len(X))
    for i in range(len(X)):
        like_0 = 1
        like_1 = 1
        for j in range(X.shape[1]):
            like_0=like_0*gaussian(X[i,j], model.m_0[j], model.std_0[j])
            like_1=like_1*gaussian(X[i,j], model.m_1[j], model.std_1[j])
        if(like_0*model.prior_0 <= like_1*model.prior_1):
            y_pred[i] = 1
    return y_pred

def Accuracy(X, y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    return (1/len(X))*sum(y_actual == y_pred)*100, matrix       
y_pred = Predictions(X_test, model)
test_accuracy, Confuse_mat = Accuracy(X_test, y_test, y_pred)
print("Test Accuracy:",test_accuracy,"%\nConfusion matrix:\n", Confuse_mat)

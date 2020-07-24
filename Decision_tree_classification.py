import numpy as np
import pandas as pd
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

############# Training model and prediction ############
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
pred_label = classifier.predict(X_test)
train_pred = classifier.predict(X_train)
def Accuracy(X, y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    return (1/len(X))*sum(y_actual == y_pred)*100, matrix       
test_accuracy, Confuse_mat = Accuracy(X_test, y_test, pred_label)
print("Test Accuracy:",test_accuracy,"%\nConfusion matrix:\n", Confuse_mat)
train_accuracy, Confuse_mat1 = Accuracy(X_train, y_train, train_pred)
print("Train Accuracy:",train_accuracy,"%\nConfusion matrix:\n", Confuse_mat1)

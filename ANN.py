import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#Importing data
dataset = pd.read_csv('Churn_Modelling.csv', header = 0)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer.fit(X[:, 0:])
X[:, 0:] = imputer.transform(X[:, 0:])

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct1 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct1.fit_transform(X))

#Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#Feature scaling(for neural networks scale all features columns)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train[1:3], X_test[1:3], y_train, y_test)

#Adding input layer
ann = tf.keras.models.Sequential()
#1st hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#2nd hidden layer
ann.add(tf.keras.layers.Dense(units = 10, activation = 'sigmoid'))
#output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#Training the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 20)


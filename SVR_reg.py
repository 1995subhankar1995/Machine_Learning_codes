import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

###########  Importing Dataset    #######################
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(X), 1)
######## Feature Scaling ###################
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)
################### Training model ###############
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y),color = 'blue')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)),color = 'red')
plt.title('SVR fit(one outlier)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

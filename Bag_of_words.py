import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


#Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#    print(review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
#Creating the bag of words models
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
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
print(sum(X_train[1]==1), sum(X_test[1]==1))


X_train_0 = X_train[np.where(y_train == 0)]
X_train_1 = X_train[np.where(y_train == 1)]
y_train_0 = y_train[np.where(y_train == 0)]
y_train_1 = y_train[np.where(y_train == 1)]
######### Calculation of mean and standard deviation ######

prior_0 = len(X_train_0)/len(X_train)
prior_1 = len(X_train_1)/len(X_train)


pi_0 = [0] * X_train_0.shape[1]
for i in range(len(X_train_0)):
    for j in range(X_train_0.shape[1]):
        if(X_train_0[i, j] > 0):
            pi_0[j] += 1
pi_1 = [0] * X_train_1.shape[1]
for i in range(len(X_train_1)):
    for j in range(X_train_1.shape[1]):
        if(X_train_1[i, j] > 0):
            pi_1[j] += 1
pi_0 = np.array(pi_0)/len(X_train_0)
pi_1 = np.array(pi_1)/len(X_train_1)

y_pred = np.zeros(len(X_test))
for i in range(len(X_test)):
    prob_0 = 1
    prob_1 = 1
    for j in range(X_test.shape[1]):
        if(X_test[i, j] > 0):
            prob_0 = prob_0 * (pi_0[j] + 0.5) #0.5 is added for Laplace smoothning
            prob_1 = prob_1 * (pi_1[j] + 0.5)
    if(prob_1 >= prob_0):
        y_pred[i] = 1

def Accuracy(X, y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    return (1/len(X))*sum(y_actual == y_pred)*100, matrix       
test_accuracy, Confuse_mat = Accuracy(X_test, y_test, y_pred)
print("Test Accuracy:",test_accuracy,"%\nConfusion matrix:\n", Confuse_mat)
print(X.shape)


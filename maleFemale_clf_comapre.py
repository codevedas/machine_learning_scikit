from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_RF = RandomForestClassifier(n_estimators=2)
clf_NB = GaussianNB()
clf_LR = LogisticRegression()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)
clf_RF.fit(X, Y)
clf_NB.fit(X, Y)
clf_LR.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_RF = clf_RF.predict(X)
acc_RF = accuracy_score(Y, pred_RF) * 100
print('Accuracy for RF: {}'.format(acc_RF))

pred_NB = clf_NB.predict(X)
acc_NB = accuracy_score(Y, pred_NB) * 100
print('Accuracy for NB: {}'.format(acc_NB))

pred_LR = clf_NB.predict(X)
acc_LR = accuracy_score(Y, pred_LR) * 100
print('Accuracy for LR: {}'.format(acc_LR))   

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN, acc_RF, acc_NB, acc_LR])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN',3:'RF',4:'NB',5:'LR'}
print('Best gender classifier is {}'.format(classifiers[index]))
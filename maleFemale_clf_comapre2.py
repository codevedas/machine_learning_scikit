from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import time

# input data [[height, weight, shoe_size]]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# known output
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron(max_iter=5 , tol=None)
clf_KNN = KNeighborsClassifier()
clf_RF = RandomForestClassifier(n_estimators=2)
clf_NB = GaussianNB()
clf_LR = LogisticRegression()

#this python dictionary {key: value}
algoList = {1: { "name": "DecisionTreeClassifier",  "model": clf_tree },
2: { "name": "SVC",  "model": clf_svm },
3: { "name": "Perceptron",  "model": clf_perceptron },
4: { "name": "KNeighborsClassifier",  "model": clf_KNN },
5: { "name": "RandomForestClassifier",  "model": clf_RF },
6: { "name": "GaussianNB",  "model": clf_NB },
7: { "name": "LogisticRegression",  "model": clf_LR }}

for number, algo in algoList.items():
	start = time.clock()   # starting th clock
	algo["model"].fit(X, Y) # training a model eg: algo["model"]==clf_tree
	prediction = algo["model"].predict(X) # testing a model with all input 
	accuracy = accuracy_score(Y, prediction) * 100  # calculating the accuracy with (expected/known output,predicted output)
	absolute_time = time.clock() - start # end time - start time (start)
	print("%s)Accuracy= %3.5f Time= %s Algorithm= %s  " % (number, accuracy, absolute_time, algo["name"] )) 
	print(algo["model"].predict([[159, 55, 37]]))  # [[159, 55, 37]] = female 
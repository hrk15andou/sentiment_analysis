# Support Vector Machine Module
# Update : 2017/01/01(Sun)

# ---- Import modules -------------------------------------- #

import numpy as np
import numpy
from sklearn import svm, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---- Learning by SVM  ------------------------------------ #

def mlearn(docs,flags,tdocs,tflgs,num):

	true = tflgs

	parameters = [{'kernel':['rbf'], 'gamma':[1e-3, 1e-4],
			'C': [1, 10, 100, 1000]},
			{'kernel':['linear'], 'C':[1,10,100,1000]}]

	svc = grid_search.GridSearchCV(svm.SVC(), parameters, cv=2, n_jobs = -1)
	svc.fit(docs,flags)
	pred = svc.predict(tdocs)

	print("Support Vector Machine")
	print(classification_report(true, pred))

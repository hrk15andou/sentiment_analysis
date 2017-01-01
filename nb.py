# Naive Bayes Module
# Update : 2017/01/01(Sun)

# ---- Import modules -------------------------------------- #

import numpy as np
import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---- Learning by NB -------------------------------------- #

def nblearn(docs,flags,tdocs,tflgs,num):

	true = tflgs

	nb = GaussianNB()
	nb.fit(docs,flags)
	nb_pred = nb.predict(tdocs)

	print("Naive Bayes")
	print(classification_report(true,nb_pred))

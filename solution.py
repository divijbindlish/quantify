from sklearn.preprocessing import Imputer

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn import cross_validation

import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import dateutil.parser as dateparser
import pickle
import os
from preprocessing import *


classifiers = {
	'knn': KNeighborsClassifier( 3 ),
	'svm_linear': SVC(kernel="linear", C=0.025),
	'svm': SVC(gamma=2, C=1),
	'tree': DecisionTreeClassifier(max_depth=5),
	'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	'adb': AdaBoostClassifier(),
	'etc': ExtraTreesClassifier(),
	'gauss': GaussianNB(),
	'mult_gauss': MultinomialNB(),
	'lda': LDA(),
	'qda': QDA()
}

def feature_selection( training_data, target_data, test_data ):
	X = np.array( training_data ).astype(np.float)
	y = np.array( target_data ).astype(np.float)

	clf = ExtraTreesClassifier()
	X_new = clf.fit( X, y ).transform( X )
	print clf.feature_importances_
	print X_new.shape
	
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_new = sel.fit_transform(X)
	print X_new.shape

	# X_new = SelectKBest( chi2, k=2 ).fit_transform( X, y )
	# print X_new.shape

	return 0

def classification():
	return 0

def cross_val( training_data, target_data, test_data ):
	train = np.array( training_data ).astype(np.float)
	target = np.array( target_data ).astype(np.float)
	# for clf_key, clf in classifiers.iteritems():
	clf_key = 'svm'
	clf = classifiers[clf_key]
	scores = cross_validation.cross_val_score( clf, train, target, cv=5 )
	print clf_key, scores.mean()

def main():
	if not os.path.exists('objects/clean_training_data.p'):
		prepare_data()

	training_data = pickle.load( open( "objects/clean_training_data.p", "r" ) )
	target_data = pickle.load( open( "objects/clean_target_data.p", "r" ) )
	test_data = pickle.load( open( "objects/clean_test_data.p", "r" ) )

	feature_selection( training_data, target_data, test_data )
	cross_val( training_data, target_data, test_data )

if __name__ == '__main__':
	main()
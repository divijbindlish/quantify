from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import dateutil.parser as dateparser
import pickle
import os
from preprocessing import *

def feature_selection( training_data, target_data, test_data ):
	X,y = training_data, target_data

	clf = ExtraTreesClassifier()
	X_new = clf.fit( X, y ).transform( X )
	print clf.feature_importances_
	print X_new.shape
	
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_new = sel.fit_transform(X)
	sel.scores_
	print X_new.shape

	# X_new = SelectKBest( chi2, k=2 ).fit_transform( X, y )
	# print X_new.shape

	return 0

def main():
	if not os.path.exists('objects/clean_training_data.p'):
		prepare_data()

	training_data = pickle.load( open( "objects/clean_training_data.p", "r" ) )
	target_data = pickle.load( open( "objects/clean_target_data.p", "r" ) )
	test_data = pickle.load( open( "objects/clean_test_data.p", "r" ) )

	feature_selection( training_data, target_data, test_data )

if __name__ == '__main__':
	main()
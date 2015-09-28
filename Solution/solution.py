from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, SelectPercentile, f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from nnet import *

from sklearn import cross_validation

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
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
	'lda': LDA(),
	'qda': QDA(),
	'ann': neuralNetwork( 16 )
}

def feature_selection( training_data, target_data, test_data ):
	X = np.array( training_data ).astype(np.float)
	y = np.array( target_data ).astype(np.float)
	X_test = np.array( test_data ).astype(np.float)
	X_index = np.arange(X.shape[-1])
	plt.figure()

	''' Variance Threshold '''
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X = sel.fit_transform( X )
	X_test = sel.transform( X_test )
	X_index = np.arange(X.shape[-1])

	''' Univariate feature selection with F-test for feature scoring '''
	sel = SelectPercentile( f_classif, percentile=10 )
	sel.fit( X, y )
	scores = -np.log10(sel.pvalues_)
	# scores /= scores.max()
	print X_index
	print scores
	plt.bar( X_index - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g' )

	''' Classifier coefficients '''
	clf = ExtraTreesClassifier()
	clf.fit( X, y )
	X = clf.transform( X )
	X_test = clf.transform( X_test )
	scores = clf.feature_importances_
	scores *= 1000
	print X.shape
	print X_test.shape
	print X_index
	print scores
	plt.bar( X_index - .25, scores, width=.2, label=r'ExtraTreesClassifier score', color='r' )
	X_index = np.arange(X.shape[-1])

	clf = classifiers['svm']
	clf.fit(X, y)
	svm_weights = (clf.coef_ ** 2).sum(axis=0)
	svm_weights /= svm_weights.max()
	plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

	''' Recursive feature elimination with cross validation '''
	estimator = classifiers['svm_linear']
	rfecv = RFECV( estimator, step=1, cv=cross_validation.StratifiedKFold(y, 2), scoring='accuracy' )
	rfecv.fit(X, y)
	X = rfecv.transform( X )
	X_test = rfecv.transform( X_test )
	scores = rfecv.grid_scores_
	scores *= 1000
	print X.shape
	print X_test.shape
	print X_index
	print scores
	plt.bar( X_index - .05, scores, width=.2, label=r'RFECV score', color='b' )

	# plt.show()

	pickle.dump( X, open( 'objects/feature_selected_training_data.p', 'wb' ) )
	pickle.dump( X_test, open( 'objects/feature_selected_test_data.p', 'wb' ) )
	return X, X_test

def classification( target_data, result_index ):
	X = pickle.load( open( "objects/feature_selected_training_data.p", "r" ) )
	y = np.array( target_data ).astype(np.float)
	X_test = pickle.load( open( "objects/feature_selected_test_data.p", "r" ) )

	for clf_key, clf in classifiers.iteritems():
		# clf_key = 'ann'
		# clf = classifiers[clf_key]
		clf.fit( X, y )
		result = clf.predict( X_test )
		new_result = []
		for r in result:
			new_result.append( 'Stripe' + str( int(r) ) )
		# print new_result
		result = pd.DataFrame( new_result, columns=['Risk_Stripe'], index=result_index)
		result.to_csv( 'result/result_' + clf_key + '.csv' )
		break

def cross_val( target_data ):
	training_data = pickle.load( open( "objects/feature_selected_training_data.p", "r" ) )
	target_data = np.array( target_data ).astype(np.float)
	for clf_key, clf in classifiers.iteritems():
		print clf_key
		scores = cross_validation.cross_val_score( clf, training_data, target_data, cv=5 )
		print scores.mean()

def main():
	if not os.path.exists('objects/clean_training_data.p'):
		prepare_data()

	training_data = pickle.load( open( "objects/clean_training_data.p", "r" ) )
	target_data = pickle.load( open( "objects/clean_target_data.p", "r" ) )
	test_data = pickle.load( open( "objects/clean_test_data.p", "r" ) )
	n_classes = len(target_data.unique())
	
	training_data, test_data = feature_selection( training_data, target_data, test_data )
	result_index = test_data.index
	classification( target_data, result_index )
	cross_val( target_data )

if __name__ == '__main__':
	main()
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
import numpy as np

class neuralNetwork():

	def __init__( self, n_classes ):
		self.n_classes = n_classes

	def fit( self, X, Y ):
		n_features = X.shape[1]
		self.train_ds = ClassificationDataSet( n_features, 1, nb_classes = self.n_classes )
		for train, target in zip( X, Y ):
			self.train_ds.addSample( train, [target] )

		self.train_ds._convertToOneOfMany( )

		self.net = buildNetwork( self.train_ds.indim, 2*n_features, self.train_ds.outdim, outclass = SoftmaxLayer )
		self.trainer = BackpropTrainer( self.net, self.train_ds )

	def predict( self, X ):
		n_features = X.shape[1]
		self.test_ds = ClassificationDataSet( n_features, 1, nb_classes = self.n_classes )
		for test in X:
			self.test_ds.addSample( test, [1] )

		self.test_ds._convertToOneOfMany( )

		for i in range( 100 ):
			self.trainer.trainEpochs( 5 )
			self.labels = self.net.activateOnDataset( self.test_ds )
			self.labels = self.labels.argmax(axis=1)
		return self.labels

if __name__ == '__main__':
	X = np.array([[ 0.,  0.],
		[ 0.,  1.],
		[ 1.,  0.],
		[ 1.,  1.]])
	Y = [ 0, 1, 1, 0 ]
	test = np.array([[ 0.,  0.],
		[ 0.,  1.]])
	n = neuralNetwork( 2 )
	n.fit( X, Y )
	print n.predict( test )
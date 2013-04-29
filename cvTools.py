import numpy as np

def kFoldCV(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	training = []
	validation = []
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in xrange(K):
		tmpTraining = [x for i, x in enumerate(X) if i % K != k]
		tmpValidation = [x for i, x in enumerate(X) if i % K == k]
		training.append(tmpTraining)
		validation.append(tmpValidation)

	return training, validation

def kRoundBS(X,k):
	""" Given a list of unique indices for data, 
	returns a list of training and validation index sets
	for bootstrapping.  This is done by selecting a 
	training set, the same size of the initial set,
	by random, uniform sampling with replacment.
	X	list of data indices 
	k 	number of bootstrap sets
	returns:
	train	list of index arrays for training
	valid	list of index arrays for validation
		corresponding to train
	""" 
	training = []
	validation = []
	n = len(X)
	for i in range(k):
		tmpTraining = map(int,sampleWR(X,n))
		uni = np.unique(tmpTraining)
		tmpValidation = np.setdiff1d(X,uni,True)
		training.append(tmpTraining)
		validation.append(tmpValidation)

	return training, validation

		
	

def sampleWR(pop,k=0):
	if k<1:k=len(pop)
	n = len(pop)
	sel = np.zeros(k)
	for i in range(k):
		index = np.random.randint(0,n)
		sel[i] = pop[index]

	return sel

class TrainError(object):
	"""A simple container for errors estimated by  
	training (ie bootstrap or crossvalidation),
	including the training error (me), 
	and the varriance for those error estimates (ve)
	Takes values as a vector for multiple models, ie,
	varriation of a tunning parameter (param).
	"""
	def __init__(self,param,nSamp,me,ve):
		"""Creates an instance of TrainError
		param	vector of a given (tunning) parameter 
		nSamp	number of samples for error estimate
		me	mean (expected) error values corrisponding to param
		ve	variance in error values 
		"""
		self._param = param
		self._me = me
		self._ve = ve
		self._nSamp = nSamp
		self._paramName = 'tunning parameter'
	
	def __len__(self):
		return len(self._param)

	@property 
	def nmodels(self):
		"""Returns the number of models 
		for which errors were calculated,
		individual models corrispond to param
		"""
		return len(self._param)

	@property
	def param(self):
		"""Parameter describing the diffrent
		models for each error contained in this 
		object.  Typically diffrent values
		of a tunning parameter for which cv 
		was conducted.
		"""
		return self._param
		
	@property	
	def mErr(self):
		"""Returns the mean (expected)
		error for each model in a vector
		which corrisponds to param.
		"""
		return self._me
	
	@property 
	def vErr(self):
		"""Returns the varriance of the
		error for each model in a vector
		which corrisponds to param.
		"""
		return self._ve

	@property
	def nSamples(self):
		"""Returns the number of samples
		for which the mean and variance were 
		estimated on.
		"""
		return self._nSamp

	def setParamName(self,paramName):
		"""Sets this objects parameter name/
		discription to the string paramName
		"""
		self._paramName = paramName

	def plotMse(self):
		"""Plots the means errors, 
		with error bars (the standard error
		given the varriance and number of samples), 
		as a function of the training parameter, param.
		"""
		import matplotlib
		import matplotlib.pyplot as plt

		plt.clf()
		interactive_state = plt.isinteractive()
	
		plt.errorbar(self._param,self._me,yerr=np.sqrt(self._ve/self._nSamp))
		plt.title('Cross Validation Error Results:')
		plt.xlabel(self._paramName)
		plt.ylabel('training error')
		plt.show()
		plt.interactive(interactive_state)
	



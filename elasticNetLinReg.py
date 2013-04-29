from glmnet import glmnet
import numpy as np
import cvTools as st


def fit(regressors, response, alpha, memlimit=None,
                largest=None, **kwargs):
	"""Preforms an elastic net constrained linear regression.
	Given the regressors matrix (row=observation,col=regressors)
	and the response vector (col=observation) an elastic net
	is fit using the alpha value (0<=value<=1; 0=ridge and 1=lasso)
	for the L1 and L2 norm constraints.  The fit is preformed via
	cyclical coordinate descent from the fortran code glmnet.f as
	described in Friedman2010.
	The output is an ElasticNetMod object that contains
	all the information on the fits for each penalty value investigated.
	"""
	lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = \
	glmnet.elastic_net(regressors, response, alpha, memlimit,
                largest, **kwargs)
	enm = ENetModel( lmu, a0, ca, ia, nin, rsq, alm, nlp,len(ca), alpha)
	return enm

def fitFull(regressors, response, alpha, nSamp=10,method='cv', 
		memlimit=None, largest=None, **kwargs):
	"""Performs samplings for error estimation to identify the best fit
	lambda, constraint penalty, and alpha, elastic net parameter,
	for a linear elastic net regression using glmnet, general linearized 
	models via coordinate decent.
	
	regressors - matrix of regressors (col=regressors,row=observations);
	response - vector of responses (col=observations);
	alpha - vector of alpha values to investigate,
	note that lambda values may be assigned but are suggested to be chosen 
	by the algorithm during optimization;
	see fitSampling for info on error estimation parameters nSamp and method,
	note that the mean squared error is used to select the best fit
	see fit for info on regression parameters.
	
	Returns an elasticNetLinMod object, enm, housing the results 
	of the best fit, the final, best, cv error is recorded in the enm.
	meanCoef	vector containing the mean value of the coef 
			for the best model over the cv, corresponding to the 
			the regressors (NOT the indices in the full model)
	varCoef		vector containing the variance of the coef 
			for the best model over the cv, corresponding to the 
			the regressors (NOT the indices in the full model)

	Note: all cv errors within one standard error (also estimated by sampling)
	are considered equivalent.  For equivalent errors the model with the 
	lowest number of non-zero regressors is considered best.
	"""
	
	# need to consider all the alpha values 	
	bestMin = 1E10
	nObs,nReg = regressors.shape
	bestReg = nReg
	for a in alpha:
		# run an enet error estimation fit to each to find best lambda  
		err, enm = fitSampling(regressors, response, a, nSamp, method, 
			memlimit, largest, **kwargs)
		
		errV = err.mErr
		stErrV = np.sqrt(err.vErr/float(nSamp))

		# find the best out of this solution:
		tmpMin = np.min(errV)
		tmpIndex = np.argmin(errV)
		tmpMinStdErr = stErrV[tmpIndex]
		tmpReg = enm.ncoefs(tmpIndex) 
		better = False
		if (tmpMin+tmpMinStdErr) < bestMin:
			# if its better by this much take it
			better = True

		elif (tmpMin < bestMin) and (tmpReg < bestReg):
			# if its better by a bit only except 
			# if there are less regressors
			better = True
	
		if better:
			# set up new bests
			bestMin = tmpMin
			bestReg = tmpReg
			bestEnm = enm[tmpIndex]
			bestMeanCoef = err.meanCoef[:,tmpIndex]
			bestVarCoef = err.varCoef[:,tmpIndex]
			# *** need the error on the null model
			# to calc COD, this is a hack 
			# but assuming that error is the first error
			# typically is (no regressors selected)
			# going to grab it and simply pass it
			nullErr = enm.errors[0]

	return bestEnm, bestMeanCoef, bestVarCoef, nullErr

				

		


def fitSampling(regressors, response, alpha, nSamp, method='cv', 
		memlimit=None, largest=None, **kwargs):
	"""Performs an elastic net constrained linear regression,
	see fit, with selected sampleing method to estimate errors
	using nSamp number of sampleings.
	methods:
	'cv'	cross validation with nSamp number of folds
	'bs'	bootstrap 
	'bs632'	boostrap 632 (weighted average of bs and training error)
	Returns a TrainingError object (cvTools) and an 
	ENetModel object for the full fit (err,enm).
	Function requires cvTools
	"""
	
	nObs,nRegs = regressors.shape
	# get the full model fit 
	fullEnm = fit(regressors, response, alpha, memlimit,
                largest, **kwargs)
	# get the lambda values determined in the full fit (going to force these lambdas for all cv's)
	lam = fullEnm.lambdas
	# the lambdas may have been user defined, don't want it defined twice 
	if kwargs.has_key('lambdas'):
		del kwargs['lambdas']

	# lets partition the data via our sampling method
	if method=='cv':
		t,v = st.kFoldCV(range(nObs),nSamp,randomise=True)
	elif (method=='bs') or (method=='bs632'):
		t,v = st.kRoundBS(range(nObs),nSamp)
	else:
		raise ValueError('Sampling method not correct')

	# lets consider many versions of errors
	# with our error being mean squared error
	# we want the epected mean squared error
	# and the corisponding variance over the diffrent versions
	nModels = len(lam)
	smse = np.zeros(nModels)
	sSqmse = np.zeros(nModels)
	# *** track the coefficent values as well
	# since spasitry can change (coef can be exactly zero in 
	# some folds but not others) we are tracking all of them
	# not good for memory
	sc = np.zeros((nRegs,nModels))
	sSqc = np.zeros((nRegs,nModels))
	# loop through the folds
	for i in range(nSamp):
		# get the training values
		X = regressors[t[i]]
		y = response[t[i]]
		enm =  fit(X, y, alpha, memlimit,
                	largest, lambdas=lam, **kwargs)
		# coef time
		sc[enm.indices,:] = sc[enm.indices,:] + enm.coef
		sSqc[enm.indices,:] = sSqc[enm.indices,:] + enm.coef**2
		# get the validation values
		Xval = regressors[v[i]]
		Yval = response[v[i]]
		nVal = float(len(Yval))
		# get the predicted responses from validation regressors
		Yhat = enm.predict(Xval)
				# what is the mean squared error?
		# notice the T was necassary to do the subtraction
		# the rows are the models and the cols are the observations
		mse = np.sum((Yhat.T-Yval)**2,1)/nVal
		# sum the rows (errors for given model)
		smse = smse + mse
		sSqmse = sSqmse + mse**2
		
	# now it is time to average and send back
	# I am putting the errors in a container 
	nSampFlt = float(nSamp)
	meanmse = smse/nSampFlt
	varmse = sSqmse/nSampFlt - meanmse**2
	if method=='bs632':
		yhat = fullEnm.predict(regressors)
		resubmse = np.sum((yhat.T-response)**2,1)/float(nObs)
		meanmse = 0.632*meanmse+(1-0.632)*resubmse
		
	mc = sc/nSampFlt
	vc = sSqc/nSampFlt - mc**2
	err = ENetTrainError(lam,nSamp,meanmse,varmse,mc,vc,alpha)
	err.setParamName('lambda')

	fullEnm.setErrors(err.mErr)
	
	return err, fullEnm 


	
		

class ENetTrainError(st.TrainError):
	"""A customized version of cvTools.TrainError to 
	contain the training results for fitting
	the lambda parameter of the elastic net.  Plotting functions
	were altered along with 2 additional slots 
	for information on regression coefficient values.
	"""
	def __init__(self,param,nSamp,me,ve,mc,vc,alpha):
		"""Creates an instance of ENetTrainError
		param	vector of the lambda parameter values trained on 
		nSamp	number of samples error was estimated on 
			(not the number of observations)
		me	expected mean squared error corrisponding to param
		ve	variance in error values
		mc	vector of mean coefficents for enet
		vc	variance of coefficents 
		alpha	the alpha parameter for this model
		"""
		super(ENetTrainError,self).__init__(param,nSamp,me,ve)
		self._alpha = alpha
		self._mc = mc
		self._vc = vc

	@property
	def alpha(self):
		"""The elastic net alpha parameter, balance"""
		return self._alpha
	
	@property 
	def meanCoef(self):
		"""The mean coefficients for each parameter investigated
		rows are the regressors, cols are the parameters"""
		return self._mc

	@property 
	def varCoef(self):
		"""The variance of the coefficients for each parameter investigated
		rows are the regressors, cols are the parameters"""
		return self._vc


	def plotMse(self):
		"""Plots the expected training errors, 
		with error bars (the standard error given 
		the varriance and the number of samples, 
		as a function of the elastic net penalty, lambda.
		"""
		import matplotlib
		import matplotlib.pyplot as plt

		plt.clf()
		interactive_state = plt.isinteractive()
		# its easier to see the plot if its in log space 
		# and we exclude the first lambda.
		xvalues = -np.log(self._param)
		# now lets plot
		plt.errorbar(xvalues[1:],self._me[1:],yerr=np.sqrt(self._ve[1:]/self._nSamp))
		plt.title('Validation Estimates For Elastic Net ($\\alpha$ = %.2f)' % self._alpha)
		plt.xlabel('$-\log(\lambda)$')
		plt.ylabel('MSE')
		plt.show()
		plt.interactive(interactive_state)

	

class ENetModel(object):
	"""General Linear, Elastic Net Model, glmnet results.
	This object contains all the information on the fits
	for each penalty value investigated. 
	"""
	def __init__(self, lmu, a0, ca, ia, nin, rsq, alm, nlp, npred, parm):
        	self._lmu = lmu
      		self._a0 = a0
        	self._ca = ca
        	self._ia = ia
        	self._nin = nin
        	self._rsq = rsq
        	self._alm = alm
        	self._nlp = nlp
        	self._npred = npred
        	self._parm = parm
		self._err = 'NA'
		self._errSet = False	

    	def __str__(self):
        	ninp = np.argmax(self._nin)
        	return ("%s object, elastic net parameter = %.3f\n" + \
                	" * %d values of lambda\n" + \
            	" * computed in %d passes over data\n" + \
            	" * largest model: %d regressors (%.1f%%), train r^2 = %.4f") % \
            	(self.__class__.__name__, self._parm, self._lmu, self._nlp,
             	self._nin[ninp], self._nin[ninp] / float(self._npred) * 100,
             	self._rsq[ninp])

    	def __len__(self):
        	return self._lmu

    	def __getitem__(self, item):
        	item = (item + self._lmu) if item < 0 else item
        	if item >= self._lmu or item < 0:
            		raise IndexError("model index out of bounds")

        	
		model =  ENetModel(
			1,
			np.array([self._a0[item]]),
			self._ca[:,item],
			self._ia,
			np.array([self._nin[item]]),
			np.array([self._rsq[item]]),
			np.array([self._alm[item]]),
			self._nlp,
			self._npred,
			self._parm	
                    	)
		if type(self._err)!= str:
			model.setErrors(np.array([self._err[item]]))

        	return model

    	@property
    	def nmodels(self):
		"""The number of models, ie penalties, considered"""
        	return self._lmu

    	@property
    	def coef(self):
		"""The matrix of coefficients for each model.  
		The rows describe the different regressors used 
		which correspond to the indices 
		and the columns are the different models which 
		correspond to the different lambdas.
		"""
		if len(self._ca.shape)>1:
        		return self._ca[:np.max(self._nin), :self._lmu]
		else:
			return self._ca[:np.max(self._nin)]

    	@property
    	def indices(self):
		"""A vector of index values which describe
		which row of the coef matrix corresponds to which 
		column of the original regressors matrix.
		""" 
        	return self._ia[:np.max(self._nin)] - 1

    	@property
    	def lambdas(self):
		"""The lambdas, or penalties, considered""" 
        	return self._alm[:self._lmu]

    	@property
    	def alpha(self):
		"""The user supplied alpha value"""
        	return self._parm

	@property
	def intercept(self):
		"""A vector of linear intercepts corrisponding to each model, 
		ie the lambdas. 
		"""
		return self._a0[:self._lmu]

	@property
	def rSq(self):
		"""The r squared values for each model
		orrisponding to the lambdas
		"""
		return self._rsq

	@property 
	def errors(self):
		"""Get the error associated with each model"""
		if self._errSet:
			return self._err
		else:
			raise LookupError("Error was not set")

	def setErrors(self,err):
		"""Set an error associated with each model 
		"""
		# lets see if this lenght jives
		try:
			if len(err)!= self._lmu:
				raise ValueError("number of error values does not match number of models")
		except:
			if self._lmu!=1:
				raise ValueError("number of error values does not match number of models")
			
		self._err = err	
		self._errSet = True	
		

	def ncoefs(self,n=-1):
		"""Returns the number of non zero coefs in 
		model n, which corresponds to the lambdas.
		The default model is the lowest penalty.
		"""
		return np.sum(np.abs(self.coef[:,n])>0)

	def predict(self, regressors):
		"""Returns a matrix of predicted responses
		given a matrix of regressors at different observations.
		All models are solved simultaneously.  the rows
		of the response matrix is an observation and the col 
		is for a specific model corresponding to lambdas.
		"""
        	regressors = np.atleast_2d(np.asarray(regressors))
        	return self.intercept + \
                	np.dot(regressors[:,self.indices], self.coef)
			


	def plot(self, which_to_label=None):
		"""Plots the regularization path,
		which is the coef values at each lambda, penalty, 
		considered.  This function only works if multiple 
		lambdas were investigated.
		Requires matplotlib
		"""
    		import matplotlib
    		import matplotlib.pyplot as plt
   	 	plt.clf()
    		interactive_state = plt.isinteractive()
    		xvalues = -np.log(self.lambdas[1:])
    		for index, path in enumerate(self.coef):
        		if which_to_label and self.indices[index] in which_to_label:
            			if which_to_label[self.indices[index]] is None:
                			label = "$x_{%d}$" % self.indices[index]
            			else:
                			label = which_to_label[self.indices[index]]
        		else:
            			label = None


        		if which_to_label and label is None:
            			plt.plot(xvalues, path[1:], ':')
        		else:
           		 	plt.plot(xvalues, path[1:], label=label)
    
    		plt.xlim(np.amin(xvalues), np.amax(xvalues)) 

    		if which_to_label is not None:
        		plt.legend(loc='upper left')
    		plt.title('Regularization paths ($\\alpha$ = %.2f)' % self.alpha)
    		plt.xlabel('$-\log(\lambda)$')
    		plt.ylabel('Value of regression coefficient $\hat{\\beta}_i$')
    		plt.show()
    		plt.interactive(interactive_state)





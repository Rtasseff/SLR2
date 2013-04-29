"""
Python2.7 module for running Sparse Linear Regression with typical settings, options and
workflows currently found most usefull.
20130118 RAT
"""
import elasticNetLinReg as enet
from glmnet import glmnet
import numpy as np
import math
import gpdPerm
import cvTools as st
from scipy.sparse import lil_matrix


def run_1(X,y,nodeNum,nodeFileName,edgeFileName):
	solution, enm = permModel(X,y,nSamp=1000,estImp=False)
	print_multi_1(solution,enm,nodeNum,nodeFileName,edgeFileName)

def print_multi_1(solution,enm,nodeNum,nodeFileName,edgeFileName):
	"""Designed for running multiple regressions in parellel
	this function prints the output of a single regression to 
	a file by appedning it.  Non Feature specific info is 
	updated to the node file (the node is this regressors 
	observable varriable, which is node name), 
	the feature specific stats will	be updated as an edge
	from node i (indicated as regressor index in solution) to 
	to node j (indicated by nodeNum).
	"""
	# Lets print the cod and intercept for the node
	cod = (solution['aveNullErr']-solution['aveErr'])/solution['aveNullErr']
	inter = enm.intercept[0]
	nodeFile = open(nodeFileName,'a')
	nodeFile.write(str(nodeNum)+'\t'+str(cod)+'\t'+str(inter)+'\n')
	nodeFile.close()
	# lets get the coef, the median, the st dev, and the pvalue
	ind = solution['indices']
	coef = solution['coef'][ind]
	med = solution['medCoef'][ind]
	sd = solution['sdCoef'][ind]
	p = solution['p'][ind]
	
	edgeFile = open(edgeFileName,'a')
	for i in range(len(ind)):
		edgeFile.write(str(ind[i])+'\t'+str(nodeNum)+'\t'+str(coef[i])+'\t'+str(med[i])+'\t'+str(sd[i])+'\t'+str(p[i])+'\n')
	edgeFile.close()
		
		



def permModel(X,y,nSamp=100,alphaList=np.array([1]),nPermsMin=1000,nPermsMax=100000,reselect=False,indType='coef', estImp=True,estErr=True):
	"""Fits the data to linear model using specified elastic net param 
	(defulat is 1, ie LASSO).  The penalty is specified by bootstrap permutations
	with nSamp (reselect determines if the penalty should be re-estimated over 
	the permutations or if the full model value should be used).  Permutations
	are done to find the permutation coef (key = 'medPermCoef') which is used 
	to estimate the p-value (key = 'p').  
	NOTE: in this version we do not calculate the standard error estimate over the 
	permutations, therfore we do not scale the coef, so the test statistic is simply
	the coefficent itself. 
	"""
	## ok this is a cut and paste job, 
	## some varriable names are not great (ie I still use the name tStat when its just
	## the abs of the coef and not really the tStat, but I think all is correct
	## in the technical sense that it does what I think it does.
	nObs,nRegs = X.shape
	solution, enm = estModel(X,y,nSamp,alphaList,indType=indType,estImp=estImp,estErr=estErr)
	medCoef = solution['medCoef']
	aveCoef = solution['aveCoef']
	sdCoef = solution['sdCoef']
	indices = solution['indices']
	
	solution['coef'] = np.zeros(nRegs)
	solution['coef'][enm.indices] = enm.coef
	lam = enm.lambdas[0]
	alpha = enm.alpha
	done = False
	p = np.ones(nRegs)
	medPermCoef = np.zeros(nRegs)
	nPerms = nPermsMin
	if len(indices)>0:
		while (nPerms<=nPermsMax and not done):
			tStat = np.zeros(nRegs)
			tStat[enm.indices] = np.abs(enm.coef)
			tStatPerm = lil_matrix((nRegs,nPerms))
			for i in range(nPerms):
				# permute the response 
				# *** probably should keep track to avoid repeats, future???
				yPerm = np.random.permutation(y)
				if reselect:
					enmPerm = select(X,yPerm,nSamp,alphaList) 
				else:
					enmPerm = enet.fit(X,yPerm,alpha,lambdas=[lam])[0]

				indPerm = enmPerm.indices
				
				
				if len(indPerm)>0:
					tmp = np.abs(enmPerm.coef)
					# more crzy shift cuz the dif from 1-d array in np and scipy
					tStatPerm[indPerm,i] = np.array(tmp,ndmin=2).T 
			#np.savetxt('tStat.dat',tStat)
			#np.savetxt('tStatPerm.dat',np.array(tStatPerm.todense()))		


				
			
			done = True 
			for i in range(nRegs):
				# even more confusion for scpy and np arrays
				# gdpPerm is expecting a vector which is diffrent 
				# from an nx1 matrix (apperently) 
				curTStatPerm = np.array(tStatPerm[i,:].todense())[0,:]
				medPermCoef[i] = np.median(curTStatPerm)
				p[i] = gpdPerm.est(tStat[i],curTStatPerm)
				# use standard permutation if this fails 
				if np.isnan(p[i]) or p[i] == 0:
					done = False
					tmp = np.sum(curTStatPerm>=tStat[i])+1
					p[i] = float(tmp)/(float(nPerms))
					if p[i]>1.0:p[i]=1.0
						
			nPerms = nPerms*10

	solution['p'] = p
	solution['medPermCoef'] = medPermCoef
	return solution, enm


def estModel(XFull,y,nSamp=100,alphaList=np.array([1]),indType='coef',estErr=True,estImp=True,reduceX=False,params=[],):
	"""Estimate a mean, median and standard deviation
	for an elastic net model using bootstrap 
	residual.
	Bootstrap resampling is used to select
	model parameters, then the bs res at these 
	params is used on the full feature set X
	to calculate the stats.  nSamp is used for
	selection and stat estimates.

	Options
	*indType* determines which stat to use for indicies.
	Indices report the non zero entries in the sparse
	regression model.  Possible types:
	coef - use coefs from full fit after the selection
	(defult)
	ave - use the avereage coefs after the bs, typically
	includes many more regressors, not recomended
	as the average removes sparsity benifit.
	med - use the median value after the bs, typically 
	fewer regressors chosen then 'coef'

	if *estErr* then 10 fold CV is used to estimate 
	the prediction error at each iteration of the bs.
	This is ten extra iterations at each bs res 
	sample, but reduces the bias in prediction error.
	The mean and sdDev of the CV error is then reported.

	If *estImp* then the importance of each selected 
	regressor is estimated.  For errOut this is the error
	if the regressor is removed, multi varriate error.
	For errIn this is the error if the regressor is alone,
	univariate error.

	If *reduceX* then the regressor matrix is ruduced 
	based on the full model fit after selection.  Only
	non zero coef are kept, much faster, but biases the 
	other stats.  
	NOTE: This was never tested after the last 
	migration, its possible the indices in the solution 
	do not match the orginal ones

	If *params* are passed then we assume its a tuple
	with the (lambda,alpha) model parameters.  In this case 
	model selection is bipassed. and these params are used.
	
	"""

	nObs,nRegsFull = XFull.shape
	# select full model values
	if len(params)==2:
		lam,alpha = params
		enm = enet.fit(XFull,y,alpha,lambdas=[lam])[0]
	else:
		enm = select(XFull,y,nSamp,alphaList)
	lam = enm.lambdas[0]
	yHat = enm.predict(XFull)
	intercept = enm.intercept[0]
	globalCoef =enm.coef[np.abs(enm.coef)>1E-21]
	coefIndex = enm.indices[np.abs(enm.coef)>1E-21]
	alpha = enm.alpha

	# now is when we reduce the x if we need too!
	if reduceX:
		nRegs = len(coefIndex)
		if nRegs > 0:
			X = XFull[:,coefIndex]
			nObs, _ = X.shape
	else:
		X = XFull
		nRegs = nRegsFull

	# get the bootstrap residual response samples
	res = y - yHat
	resCent = res-np.mean(res)
	ySample = np.zeros((nObs,nSamp))
	for i in range(nSamp):
		resSample = st.sampleWR(resCent)
		ySample[:,i] = yHat+resSample

	if nRegs > 0:
	
		# residual bs time
		if estErr:
			sumErr = 0
			sumSqErr = 0
			sumNullErr = 0
			sumSqNullErr = 0

		sc = np.zeros(nRegs)
		sSqc = np.zeros(nRegs)
		ac = lil_matrix((nRegs,nSamp))
		sumSup = np.zeros(nRegs)
		

		for i in range(nSamp):
			# cv to get the errors
			if estErr:
				err,tmpEnm,tmpallVals = fitSampling(X,ySample[:,i],alpha,10,method='cv',lambdas=[lam])
				sumErr = err.mErr[0] + sumErr
				sumSqErr = err.mErr[0]**2 + sumSqErr
				# cv over this thing to get the null model errors
				nullErr,a = fitSamplingNull(ySample[:,i],10, method='cv')
				sumNullErr = sumNullErr + nullErr
				sumSqNullErr = sumSqNullErr + nullErr**2

			# need the coef
			# they change so we need to map the back to the original
			tmpEnm = enet.fit(X,ySample[:,i], alpha,lambdas=[lam])
			sc[tmpEnm.indices] = sc[tmpEnm.indices] + tmpEnm.coef[:,0]
			sSqc[tmpEnm.indices] = sSqc[tmpEnm.indices] + tmpEnm.coef[:,0]**2
			if len(tmpEnm.indices)>0:
				ac[tmpEnm.indices,i] = tmpEnm.coef
			# find supports 
			occur = np.zeros(len(tmpEnm.coef[:,0]))
			occur[abs(tmpEnm.coef[:,0])>1E-25] = 1.0
			sumSup[tmpEnm.indices] = sumSup[tmpEnm.indices] + occur
				

		# get averages and variances
		if estErr:
			aveErr = sumErr/nSamp
			sdErr = np.sqrt(sumSqErr/nSamp - aveErr**2)
			aveNullErr = sumNullErr/nSamp
			sdNullErr = np.sqrt(sumSqNullErr/nSamp - aveNullErr**2)

		aveCoef = sc/nSamp
		sdCoef = np.sqrt(sSqc/nSamp - aveCoef**2)
		#some crazy stuff here becase of the way scipy mat is shaped
		medCoef = np.array(np.median(ac.todense(),1))[:,0]
		pSup = sumSup/nSamp

		# lets do the selection 
		if indType=='coef':
			indices = coefIndex
		elif indType=='med':
			indices = np.arange(nRegs)[np.abs(medCoef)>1E-21]
		elif indType=='ave':
			indices = np.arange(nRegs)[np.abs(aveCoef)>1E-21]
		else:
			raise ValueError('The indType '+indType+' is not valid.')

		# put it in a dict for simplicity 
		solution = {}
		if estErr:
			solution['aveErr'] = aveErr
			solution['sdErr'] = sdErr
			solution['aveNullErr'] = aveNullErr
			solution['sdNullErr'] = sdNullErr
		if reduceX:
			# need to go back to the original indicies 
			solution['aveCoef'] = np.zeros(nRegsFull)
			solution['sdCoef'] = np.zeros(nRegsFull)
			solution['medCoef'] = np.zeros(nRegsFull)
			solution['pSup'] = np.zeros(nRegsFull)

			solution['aveCoef'][coefIndex] = aveCoef
			solution['sdCoef'][coefIndex] = sdCoef
			solution['medCoef'][coefIndex] = medCoef
			solution['pSup'][coefIndex] = pSup
			solution['indices'] = coefIndex[indices]
		else:
			solution['aveCoef'] = aveCoef
			solution['sdCoef'] = sdCoef
			solution['medCoef'] = medCoef
			solution['pSup'] = pSup
			solution['indices'] = indices
		
		nRegsHat = len(indices)
		if nRegsHat>0 and estImp:
			Xhat = X[:,indices]
			# lets do the leave one out importance deal
			errOutHat = np.zeros(nRegsHat) 
			if nRegsHat>1:
				for j in range(nRegsHat):
					Xprime = np.delete(Xhat,j,axis=1)

					# residual bs time
					sumErr = 0
					sumSqErr = 0
					
					for i in range(nSamp):
						# cv to get the errors
						err,tmpenm,tmpallVals = fitSampling(Xprime,ySample[:,i],alpha,10,method='cv',lambdas=[lam])
						sumErr = err.mErr[0] + sumErr
						sumSqErr = err.mErr[0]**2 + sumSqErr

					errOutHat[j] = sumErr/nSamp

			elif nRegsHat==1:
				errOutHat[0] = aveNullErr

			# lets do leave only one
			errInHat = np.zeros(nRegsHat) 
			for j in range(nRegsHat):
				Xprime = np.zeros((nObs,1))
				Xprime[:,0] = Xhat[:,j]

				# residual bs time
				sumErr = 0
				sumSqErr = 0
				
				for i in range(nSamp):
					# cv to get the errors
					err,tmpenm,tmpallVals = fitSampling(Xprime,ySample[:,i],alpha,10,method='cv',lambdas=[lam])
					sumErr = err.mErr[0] + sumErr
					sumSqErr = err.mErr[0]**2 + sumSqErr

				errInHat[j] = sumErr/nSamp

			errOut = np.zeros(nRegs)
			errOut[indices] = errOutHat
			solution['errOut'] = errOut
			errIn = np.zeros(nRegs)
			errIn[indices] = errInHat
			solution['errIn'] = errIn



	else:
			solution = {}
			if estErr:
				sumNullErr = 0
				sumSqNullErr = 0
			
				for i in range(nSamp):
					# cv over this thing to get the null model errors
					nullErr,a = fitSamplingNull(ySample[:,i],10, method='cv')
					sumNullErr = sumNullErr + nullErr
					sumSqNullErr = sumSqNullErr + nullErr**2
			
				# get averages and variances
				aveNullErr = sumNullErr/nSamp
				sdNullErr = np.sqrt(sumSqNullErr/nSamp - aveNullErr**2)
				aveErr = aveNullErr
				sdErr = sdNullErr
				solution['aveErr'] = aveErr
				solution['sdErr'] = sdErr
				solution['aveNullErr'] = aveNullErr
				solution['sdNullErr'] = sdNullErr

			solution['aveCoef'] = np.zeros(nRegsFull)
			solution['sdCoef'] = np.zeros(nRegsFull)
			solution['medCoef'] = np.zeros(nRegsFull)
			solution['pSup'] = np.zeros(nRegsFull)
			solution['indices'] = np.array([])

		
	
			

	return solution, enm 


def select(X,y,nSamp=100,alphaList=np.array([1])):
	"""Select an elastic net model based 
	on a resampling method and return that
	model.
	"""
	nObs,nRegs = X.shape
	sdY = np.sqrt(np.var(y))
	# selection via bootstrap
	bestMin = 1E10
	for a in alphaList:
		tmpErr,tmpEnm,allVals = fitSampling(X,y,a,nSamp,method='bs')
		tmpErrV = tmpErr.mErr
		tmpMin = np.min(tmpErrV)
		
		if tmpMin < bestMin:
			bestMin = tmpMin
			modelIndex = np.argmin(tmpErrV)
			enm = tmpEnm
			err = tmpErr
			alpha = a
	
	# important values
		
	return enm[modelIndex]

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
	fullEnm = enet.fit(regressors, response, alpha, memlimit,
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
	allVals = np.zeros((nModels,nSamp))

		# loop through the folds
	for i in range(nSamp):
		# get the training values
		X = regressors[t[i]]
		y = response[t[i]]
		enm =  enet.fit(X, y, alpha, memlimit,
                	largest, lambdas=lam, **kwargs)
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
		allVals[:,i] = mse
		
	# now it is time to average and send back
	# I am putting the errors in a container 
	nSampFlt = float(nSamp)
	meanmse = smse/nSampFlt
	varmse = sSqmse/nSampFlt - meanmse**2
	if method=='bs632':
		yhat = fullEnm.predict(regressors)
		resubmse = np.sum((yhat.T-response)**2,1)/float(nObs)
		meanmse = 0.632*meanmse+(1-0.632)*resubmse
		
	err = enet.ENetTrainError(lam,nSamp,meanmse,varmse,[0],[0],alpha)
	err.setParamName('lambda')

	fullEnm.setErrors(err.mErr)
	
	return err, fullEnm, allVals 

def fitSamplingNull(response,nSamp, method='cv', 
		memlimit=None, largest=None, **kwargs):
	nObs = len(response)
	# lets partition the data via our sampling method
	if method=='cv':
		t,v = st.kFoldCV(range(nObs),nSamp,randomise=True)
	elif (method=='bs') or (method=='bs632'):
		t,v = st.kRoundBS(range(nObs),nSamp)
	else:
		raise ValueError('Sampling method not correct')

	smse = 0
	sSqmse = 0

	for i in range(nSamp):
		# get the training values
		
		y = response[t[i]]
		Yval = response[v[i]]
		nVal = float(len(Yval))
		mse = np.sum((Yval-np.mean(y))**2)/nVal
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
		
	return meanmse, varmse



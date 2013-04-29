import numpy as np
import math
import gpdPerm

def olsTTestPermute(regressors,response,nperm=1000):
	"""Caclulates p (significance) values for the 
	regressors in an ordinarly least squares fit, 
	null assupmtion is that the regressor 
	coefficent is zero.  Calculates t statistic and
	performs a permutation test to; applie a generalized 
	pereto dist to approximate t-stat distribution tail when
	appropriate.
	
	regressors - matrix of regression varriables 
	(col-regressors row-observation)
	response - vector of the response varriable (col-observation)
	nperm - scalar number of permutations

	returns
	p 	values corrisponding to the col of 
		regressors.
	tStat	the test statistic 
	tStatPerm	tStats for random permutations
			the rows - regressorsm the col - permutations
	coef	the coefficents from the linear fit
	"""
	n,m = regressors.shape
	# check to see if we have enough observations
	if math.factorial(n)<nperm:
		raise ValueError("Not enough observations \
			for {} permutations".format(nperm))

	
	# get the OLS coef estimates:
	coefs,a,b,c = np.linalg.lstsq(regressors,response) 
	yHat = np.dot(regressors,coefs)
	# get the sum of the sum residuals squared
	srs = np.sum((response.T-yHat)**2)
	# calculate the co square inverse 
	cInv = np.linalg.inv(np.dot(regressors.T,regressors))
	# coef error estimates
	d = np.diag(cInv)
	s = np.sqrt(np.abs((1.0/(n-1))*srs*d))
	# t-statistic
	tStat = np.abs(coefs)/s
	tStatPerm = np.ones((m,nperm))
	for i in range(nperm):
		# permute the response 
		# *** probably should keep track to avoid repeats, future???
		responsePerm = np.random.permutation(response)
		# repeat calc of tStat
		coefsPerm,a,b,c = np.linalg.lstsq(regressors,responsePerm)
		yHat = np.dot(regressors,coefsPerm)
		srs = np.sum((responsePerm.T-yHat)**2)
		# no need to redo the operations on regressor matrix
		s = np.sqrt(np.abs((1.0/(n-1))*srs*d))
		tStatPerm[:,i] = np.abs(coefsPerm)/s

	p = np.ones(m)*2
	for i in range(m):
		p[i] = gpdPerm.est(tStat[i],tStatPerm[i,:])

	return p, tStat, tStatPerm, coefs


def netTTestPermute(regressors,response,lam,alpha,nperm=1000):
	"""Caclulates p (significance) values for the 
	regressors in an elastic net linear fit, 
	null assupmtion is that the regressor 
	coefficent is zero.  Calculates t statistic and
	performs a permutation test to; applie a generalized 
	pereto dist to approximate t-stat distribution tail when
	appropriate.
	
	regressors - matrix of regression varriables 
	(col-regressors row-observation)
	response - vector of the response varriable (col-observation)
	lam - scalar float; elastic net lambda (penalty) parameter
	alpha - scalar float; elastic net alpha (balance) parameter
	nperm - scalar number of permutations

	returns
	p 	values corrisponding to the col of 
		regressors.
	tStat	the test statistic 
	tStatPerm	tStats for random permutations
			the rows - regressorsm the col - permutations
	coef	the coefficents from the linear fit
	"""
	import elasticNetLinReg as enet
	
	n,m = regressors.shape
	# check to see if we have enough observations
	if math.factorial(n)<nperm:
		raise ValueError("Not enough observations \
			for {} permutations".format(nperm))

	
	# get the enet coef estimates:
	coefs = np.zeros(m)
	enm = enet.fit(regressors,response,alpha,lambdas=[lam])
	coefs[enm.indices] = enm.coef
	#*********
	yHat = enm.predict(regressors)
	
	# get the sum of the sum residuals squared
	srs = np.sum((response.T-yHat)**2)
	# calculate the co square inverse 
	cInv = np.linalg.inv(np.dot(regressors.T,regressors))
	# coef error estimates
	d = np.diag(cInv)
	s = np.sqrt(np.abs((1.0/(n-1))*srs*d))
	#*********
	# t-statistic
	tStat = np.abs(coefs)/s
	tStatPerm = np.ones((m,nperm))
	for i in range(nperm):
		# permute the response 
		# *** probably should keep track to avoid repeats, future???
		responsePerm = np.random.permutation(response)
		# repeat calc of tStat
		coefsPerm = np.zeros(m)
		enmPerm = enet.fit(regressors,responsePerm,alpha,lambdas=[lam])
		coefsPerm[enmPerm.indices] = enmPerm.coef
		yHat = enmPerm.predict(regressors)
		srs = np.sum((responsePerm.T-yHat)**2)
		# no need to redo the operations on regressor matrix
		sPerm = np.sqrt(np.abs((1.0/(n-1))*srs*d))
		tStatPerm[:,i] = np.abs(coefsPerm)/sPerm

	p = np.ones(m)*2
	for i in range(m):
		p[i] = gpdPerm.est(tStat[i],tStatPerm[i,:])

	return p, tStat, tStatPerm, s


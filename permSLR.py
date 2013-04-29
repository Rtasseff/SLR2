#####
# run the response premutation 
# for the SLR model in SLR2.py
# to obtain st errors on coeff
# for the null distribution 
# and finally calc the p-values 
# 
#####


import elasticNetLinReg as enet
from glmnet import glmnet
import numpy as np
import math
import gpdPerm
import cvTools as st


def run(X,y,name,nPerms=1000):
	n,m = X.shape
	# check to see if we have enough observations
	if math.factorial(n)<nPerms:
		raise ValueError("Not enough observations \
			for {} permutations".format(nPerms))

	# open preexisting model file
	f = open('SLR2run_'+name+'.dat','r')
	# get node properties 
	line = f.next()
	lam = line.split()[0]

	line = f.next()
	alpha = line.split()[0]

	line = f.next()
	intercept = line.split()[0]

	line = f.next()
	aveErr = line.split()[0]

	line = f.next()
	sdErr = line.split()[0]

	line = f.next()
	aveNullErr = line.split()[0]

	line = f.next()
	sdNullErr = line.split()[0]
	
	line = f.next()
	sdY = line.split()[0]
	
	# look to see if there are indices (non zero coef)
	try:
		line = f.next()
		indices = line.split()

		line = f.next()
		sdX = line.split()
		
		line = f.next()
		coefs = line.split()

		line = f.next()
		meanCoef = line.split()

		line = f.next()
		sdCoef = line.split()

		line = f.next()
		pSup = line.split()

		line = f.next()
		errOut = line.split()
	
		line = f.next()
		errIn = line.split()


	except:
		# if not, set to empty
		indices = []
	
	# done with that file
	f.close()
	# ok we are doing a bit of a cheat here
	# only looking at non zero coefs from original 
	# model, should redo the entier selection process
	if len(indices)>0:
		coefIndex = np.array(map(int,indices))
		# working on subset now
		Xhat = X[:,coefIndex]
		nObs,nRegsHat = Xhat.shape

		# t-statistic
		fltSd = np.array(map(float,sdCoef))
		fltSd[fltSd<1E-52]=1E-52
		tStat = np.array(map(float,meanCoef))/fltSd
		tStatPerm = np.ones((nRegsHat,nPerms))
		for i in range(nPerms):
			# permute the response 
			# *** probably should keep track to avoid repeats, future???
			yPerm = np.random.permutation(y)
			# calc of tStat
			tStatPerm[:,i] = getTStat(Xhat,yPerm,float(alpha),float(lam))
	
		# no values should have 2 in the end,
		# this will let us know if something goes wrong
		p = np.ones(nRegsHat)*2
		for i in range(nRegsHat):
			p[i] = gpdPerm.est(tStat[i],tStatPerm[i,:])
			# use standard permutation if this fails 
			if np.isnan(p[i]) or p[i] < 1E-52:
				z = tStatPerm[i,:]
				tmp = np.sum(z>tStat[i])+1
				p[i] = float(tmp)/(float(nPerms))

		

	# we have it all, lets print it
	f = open('SLR2run_perm_'+name+'.dat','w')
	
	f.write(lam+"\n")
	
	f.write(alpha+"\n")
	
	f.write(intercept+"\n")	

	f.write(aveErr+"\n")

	f.write(sdErr+"\n")

	f.write(aveNullErr+"\n")

	f.write(sdNullErr+"\n")
	
	f.write(sdY+"\n")
	
	if len(indices)>0:
		np.array(indices).tofile(f,sep="\t")
		f.write("\n")
	
		np.array(sdX).tofile(f,sep="\t")
		f.write("\n")

		np.array(coefs).tofile(f,sep="\t")
		f.write("\n")
		
		np.array(meanCoef).tofile(f,sep="\t")
		f.write("\n")

		np.array(sdCoef).tofile(f,sep="\t")
		f.write("\n")

		np.array(pSup).tofile(f,sep="\t")
		f.write("\n")

		np.array(errOut).tofile(f,sep="\t")
		f.write("\n")

		np.array(errIn).tofile(f,sep="\t")
		f.write("\n")
	
		p.tofile(f,sep="\t")
		f.write("\n")



	f.close()



			


def getTStat(X,y,alpha,lam,nSamp=100):
	# here we are doing residual bootstrap 
	# to identify the std err and report 
	# the t-stat (mean/st err)
	nObs,nRegs = X.shape
	# sd is done by res boot so we need to get the res
	enm = enet.fit(X,y, alpha,lambdas=[lam])
	yHat = enm.predict(X)[:,0]
	res = y - yHat
	resCent = res-np.mean(res)
	ySample = np.zeros((nObs,nSamp))
	# now we need the samples 
	for i in range(nSamp):
		resSample = st.sampleWR(resCent)
		ySample[:,i] = yHat+resSample

	# residual bs time
	sc = np.zeros(nRegs)
	sSqc = np.zeros(nRegs)

	for i in range(nSamp):
		# need the coef
		# they change so we need to map the back to the original
		tmpEnm = enet.fit(X,ySample[:,i], alpha,lambdas=[lam])
		sc[tmpEnm.indices] = sc[tmpEnm.indices] + tmpEnm.coef[:,0]
		sSqc[tmpEnm.indices] = sSqc[tmpEnm.indices] + tmpEnm.coef[:,0]**2
			

	# get averages and variances
	aveCoef = sc/float(nSamp)
	sdCoef = np.sqrt(sSqc/float(nSamp) - aveCoef**2)

	# get tstat
	# due to the sparsity of lasso
	# its possible for a coef to be zero 
	# on all samples, thus a zero st error
	# we are going to remove the zeros
	sdCoef[sdCoef<1E-52] = 1E-52
	
	tStat = np.abs(aveCoef/sdCoef)
	
	return tStat




	
	


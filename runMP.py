from dispatch import dispatcher, smp
import numpy as np
# import the run modual
import SLRFull

# load the regressor matrix
#X = np.loadtxt('X.dat',delimiter="\t")
#Y = np.loadtxt('Y.dat',delimiter="\t")

# diffrent delimination
X = np.loadtxt('th.dat')
Y = np.loadtxt('dth.dat')

n,m = Y.shape

disp = smp.SMPDispatcher(m)
for i in range(1):
	y = Y[:,i]

	#need to do a tiny bit of manipulation here 
	Xhat = np.sin(X.T - X[:,i]).T
	name = 'SLR_Lasso_Orlando2008_'+str(i)+'.dat'

	args = (X,y,name,100,0)
	job = dispatcher.Job(SLRFull.run,args)
	disp.add_job(job)

disp.dispatch()

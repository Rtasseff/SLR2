from dispatch import dispatcher, smp
import numpy as np
# import the run modual
#import SLR2 
import SLRFull

# load the regressor matrix
#X = np.loadtxt('X.dat',delimiter="\t")
#Y = np.loadtxt('Y.dat',delimiter="\t")

# diffrent delimination
X = np.loadtxt('th.dat')
Y = np.loadtxt('dth.dat')


n,m = Y.shape


disp = smp.SMPDispatcher(1)
i = 4
y = Y[:,i]

#need to do a tiny bit of manipulation here 
Xhat = np.sin(X.T - X[:,i]).T

args = (Xhat,y,'test_run_'+str(i)+'.dat')
job = dispatcher.Job(SLRFull.run,args)
disp.add_job(job)

disp.dispatch()

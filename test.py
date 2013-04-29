import numpy as np
import SLR2
import permSLR_v2
name = 'test3'
X = np.random.randn(30,50)
b = np.random.randn(50)
b[5:] = 0
b[0] = b[0]*10
b[2] = b[2]*.5
b[3] = b[3]*.1
b[4] = b[4]*.01
y = np.dot(X,b) + np.random.randn(30)*.1
slr = SLR2.SLR(X,y)
slr.fit()
slr.estStErr()
slr.estImp()
slr.save(name)
print b[:6]
permSLR_v2.run(X,y,name,nPerms=10000)


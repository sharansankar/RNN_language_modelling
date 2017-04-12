import time;
import scipy as sp;
import scipy.linalg as la;
import numpy as np;
from numba import jit

n=1000
A=sp.random.rand(n,n)
B=sp.random.rand(n,n)

def numpy_dot2():
    t=time.time()
    C=np.dot(A,B)
    t=time.time()-t
    f=2*n**3/t/1e9
    print("Numpy dot2:   time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))



@jit
def numpy_dot():
    t=time.time()
    C=np.dot(A,B)
    t=time.time()-t
    f=2*n**3/t/1e9
    print("Numpy dot:   time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))
@jit
def scipy_dot():
    t=time.time()
    C=sp.dot(A,B)
    t=time.time()-t
    f=2*n**3/t/1e9
    print("Scipy dot:   time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))

@jit
def scipy_dgemm():
    t=time.time()
    C=la.blas.dgemm(1.0,A,B)
    t=time.time()-t
    f=2*n**3/t/1e9
    print("Scipy dgemm: time = %.2f seconds; flop rate = %.2f Gflops/s"%(t,f))

numpy_dot()
numpy_dot2()
numpy_dot()
numpy_dot()
numpy_dot()



#scipy_dot()
#scipy_dgemm()

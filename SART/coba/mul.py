import numpy as np
import os
from timeit import timeit
import time
from multiprocessing import Pool

def fib(n):
    if n<=1:
        return 1
    else:
        return fib(n-1)+fib(n-2)

def silly_mult(matrix):
    for row in matrix:
        for val in row:
            val * val

if __name__ == '__main__':


    # dt = timeit(lambda: map(fib, xrange(10)), number=10)
    # print "Fibonacci, non-parallel: %.3f" %dt
    print map(fib, xrange(4))

    matrices = [np.random.randn(1000,1000) for ii in xrange(10)]
    dt = timeit(lambda: map(silly_mult, matrices), number=10)
    print "Silly matrix multiplication, non-parallel: %.3f" %dt

    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all CPUS
    os.system("taskset -p 0xff %d" % os.getpid())

    pool = Pool(8)
    st = time.time()
    pool.map(fib(xrange(10)))
    end = time.time()

    # dt = timeit(lambda: pool.map(fib,xrange(10)), number=10)
    print "Fibonacci, parallel: %.3f" %(end-st)

    dt = timeit(lambda: pool.map(silly_mult, matrices), number=10)
    print "Silly matrix multiplication, parallel: %.3f" %dt
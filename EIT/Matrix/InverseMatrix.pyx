cimport numpy as np
import scipy.linalg as la
import numpy as np
from cython.parallel import prange

ctypedef np.float64_t REAL
ctypedef np.int64_t LONG
ctypedef np.complex COMPLEX

import cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
cpdef np.ndarray[COMPLEX, ndim=2] inverse(np.ndarray[COMPLEX, ndim=2] mat):
    cdef np.ndarray[COMPLEX, ndim=2] olah = np.zeros(shape=(2*len(mat),2*len(mat)), dtype=np.complex)
    cdef np.ndarray[COMPLEX, ndim=2] hasil = np.zeros(shape=(len(mat),len(mat)), dtype=np.complex)

    cdef np.ndarray[COMPLEX, ndim=2] hasilLib = np.zeros(shape=(len(mat),len(mat)), dtype=np.complex)

    cdef int num_threads=4;

    # for thread in prange(num_threads, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
    # hasilLib = la.inv(mat)

    cdef long bar = len(mat)
    cdef long kol = len(mat[0])
    cdef long i = 0
    cdef long j = 0
    cdef long k = 0
    cdef long l = 0
    cdef COMPLEX d = 0
    cdef long N = bar

    for i in range(0,N):
        for j in range(0,N):
            olah[i][j]=mat[i][j]

    for i in range(0,N):
        for j in range(0,2*N):
            if(j==(i+N)):
                olah[i][j]=1

    # print olah

    for i in range(N-1,0,-1):
        if(olah[i-1][0]<olah[i][0]):
            for j in range(2*N):
                d=olah[i][j]
                olah[i][j]=olah[i-1][j]
                olah[i-1][j]=d

    for i in range(0,N):
        for j in range(0,2*N):
            if(j!=i):
                d=olah[j][i]/olah[i][i]
                for k in range(0,2*N):
                    olah[j][k] = olah[j][k] - olah[i][k]*d

    for i in range(0,N):
        d=olah[i][i]
        for j in range(0,2*N):
            olah[i][j]=olah[i][j]/d

    for i in range(0,N):
        l=0
        for j in range(N,2*N,1):
            hasil[i][l] = olah[i][j]
            l=l+1

    return hasil

@cython.boundscheck(False)
cpdef np.ndarray[REAL, ndim=2] kali(np.ndarray[REAL, ndim=2] X, np.ndarray[REAL, ndim=2] Y):
    cdef np.ndarray[REAL, ndim=2] hasil = np.zeros(shape=(len(X[0]),len(Y[1])), dtype=np.float64)
    cdef long bar=len(X[0])
    cdef long kol=len(Y[1])
    cdef long teng=len(X[1])

    for i in range(bar):
        for j in range(kol):
            for k in range(teng):
                hasil[i][j] = hasil[i][j]+(X[i][k]*Y[k][j])

    return hasil

# import cython
# from cython.parallel import prange, parallel

# @cython.boundscheck(False)
# def slope_cython_openmp(double [:, :] indata, double [:, :] outdata):
#     cdef int I, J
#     cdef int i, j, x
#     cdef double k, slp, dzdx, dzdy
#     I = outdata.shape[0]
#     J = outdata.shape[1]
#     with nogil, parallel(num_threads=4):
#         for i in prange(I, schedule='dynamic'):
#             for j in range(J):
#                 dzdx = (indata[i+1, j] - indata[i+1, j+2]) / 2
#                 dzdy = (indata[i, j+1] - indata[i+2, j+1]) / 2
#                 k = (dzdx * dzdx) + (dzdy * dzdy)
#                 slp = k**0.5 * 100
#                 outdata[i, j] = slp
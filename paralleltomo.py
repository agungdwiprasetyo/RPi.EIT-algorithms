#!/usr/bin/python3

# """ Author : Agung Dwi Prasetyo """

import numpy as np
import matplotlib as plot
import multiprocessing as mp

# untuk fungsi paralel tomo, nilai default yaitu N=80, teta=0 sampai 179, p=None, d=None, isDisp=None
def paralleltomo(N=80,teta=list(range(0,179)),p=None,d=None,isDisp=0):
    if not p:
        p = round(np.sqrt(2)*N)
    if not d:
        d = np.sqrt(2)*N

    # menghitung panjang dari nilai teta
    nA = teta.__len__()

    # membuat rentang nilai x0
    x0 = np.linspace(-d/2,d/2,p)
    y0 = np.zeros(p)

    x = np.arange(-N,N+1)
    y = x

    #
    rows = np.zeros(2*nA*N*p)
    cols = rows
    vals = rows
    idxend = 0

    if isDisp:
        # bangkitkan matriks berukuran N*N yag nilai elemennya random
        AA = np.random.uniform(0,1,N*N) # membangkitkan bilangan random dari 0 sampai 1, masih berupa array 1 dimensi
        AA = np.reshape(AA,(N,N)) # mengubah array 1 dimensi menjadi array 2 dimensi

    # for i in range(1,nA):
    #    if isDisp:
            #
        #

    # masih coba-coba
    A = 1
    b = 1
    xx = x
    teta = x0.__len__()
    return A,b,xx,teta

N = 80
teta = list(range(0,180,5))
print(paralleltomo(80,teta))
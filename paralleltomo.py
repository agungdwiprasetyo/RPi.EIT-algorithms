""" Author : Agung Dwi Prasetyo """

import numpy as np

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

    A = 1
    b = 1
    xx = x
    teta = x0.__len__()
    return A,b,xx,teta

N = 80
teta = list(range(0,180,5))
print(paralleltomo(80,teta))
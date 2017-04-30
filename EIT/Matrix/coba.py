from InverseMatrix import inverse
from InverseMatrix import kali
import numpy as np
import scipy.linalg as la
from time import time

A=np.array([[1,2,4,3,4,3,7,8,1,2,6],
			[9,5,4,7,2,3,4,1,2,3,6],
			[1,5,4,9,5,9,1,0,0,2,3],
			[1,1,3,2,2,3,4,5,2,3,4],
			[1,1,2,7,2,3,4,5,2,3,4],
			[1,1,2,2,9,3,4,5,2,3,4],
			[1,1,2,2,2,4,4,5,2,3,4],
			[1,1,2,2,2,3,9,3,2,3,4],
			[1,1,2,2,2,3,4,4,2,3,4],
			[1,1,2,2,2,3,4,6,2,3,4],
			[1,1,2,2,2,3,4,1,2,3,4]],dtype=np.float64)
# A=np.array([[1,2,3,4,6],[0,1,4,0,3],[5,6,0,3,3],[4,3,3,4,4],[0,0,2,1,9]],dtype=np.float64)

print(np.diag(A))
# print("\n")
# cc=time()
# Z=inverse(A)
# print("\n")
# print(Z)
# print(time()-cc)

print("\ninverse lib")
xx=time()
S=la.inv(A)
print(S)
print(time()-xx)

print ("\n")
print ("kali lib:")
xx=time()
a1 = np.dot(S,A)
print(np.array(a1,dtype=np.int64))
print(time()-xx)

print ("\n")
print ("kali cython:")
xx=time()
a2 = kali(S,A)
print(np.array(a2,dtype=np.int64))
print(time()-xx)
# for i in range(len(a2)):
# 	print(a2[i])

# B=np.array([1,2,3],dtype=np.int32)
# x=IM.sum_plus_one(B)
# print(x)

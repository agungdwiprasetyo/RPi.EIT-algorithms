from Matrix.InverseMatrix import inverse
import numpy as np
import scipy.linalg as la
from time import time

A=np.array([[1,2,4,3,4,3,7,8,1,2,6,8,9,0],
			[9,5,4,7,2,3,4,1,2,3,6,7,3,1],
			[1,5,4,9,5,9,1,0,0,2,3,4,5,1],
			[1,1,3,2,2,3,4,5,2,3,4,5,6,3],
			[1,1,2,7,2,3,4,5,2,3,4,5,6,3],
			[1,1,2,2,9,3,4,5,2,3,4,5,6,3],
			[1,1,2,2,2,4,4,5,2,3,4,5,6,3],
			[1,1,2,2,2,3,9,3,2,3,4,5,6,3],
			[1,1,2,2,2,3,4,4,2,3,4,5,6,3],
			[1,1,2,2,2,3,4,6,2,3,4,5,6,3],
			[1,1,2,2,2,3,4,1,2,3,4,5,6,3],
			[1,1,2,2,2,3,4,3,2,3,4,5,6,3],
			[1,1,2,2,2,3,4,8,2,3,4,5,6,3],
			[1,5,4,9,2,2,1,0,0,2,3,4,5,1],],dtype=np.complex)
# A=np.array([[1,2,8],[9,5,4],[1,5,6]],dtype=np.complex)

print(A)

cc=time()
Z=inverse(A)
print(Z)
print(time()-cc)

print("lib")
aa=time()
S=la.inv(A)
print(S)
print(time()-aa)

# B=np.array([1,2,3],dtype=np.int32)
# x=IM.sum_plus_one(B)
# print(x)

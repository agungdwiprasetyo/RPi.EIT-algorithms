from __future__ import division, absolute_import, print_function

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import multiprocessing as mp
# import numba as nb
# from numba import jit, autojit
# from .Matrix import InverseMatrix as IM
from time import time

import multiprocessing.pool
class NoDaemonProcess(multiprocessing.Process):
	# make 'daemon' attribute always return False
	def _get_daemon(self):
		return False
	def _set_daemon(self, value):
		pass
	daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
	Process = NoDaemonProcess

class Forward(object):
	def __init__(self, mesh, elPos, useMultiprocessing=False):
		self.nodeXY = mesh['node']
		self.element = mesh['element']
		self.triPerm = mesh['alpha']
		self.elPos = elPos
		self.noNum, self.dim = self.nodeXY.shape
		self.elNum, self.nVertices = self.element.shape
		self.useMultiprocessing = useMultiprocessing

	def solve(self, exMat=None, step=1, perm=None, parser=None):
		pool = mp.Pool(processes=mp.cpu_count())
		self.step=step
		self.parser=parser
		self.exMat=exMat

		if perm is not None:
			triPerm = perm
		else:
			triPerm = self.triPerm

		numLines = np.shape(exMat)[0]
		self.numlines = numLines

		output = ['jac', 'v', 'b_matrix']
		jacobian, v, b_matrix = [], [], [] # output

		if self.useMultiprocessing:
			parallel = pool.map(self.solveJacMp, range(numLines))	# 24.768184900283813
			for z in parallel:
				v.append(z[0])
				jacobian.append(z[1])
				b_matrix.append(z[2])
		else:
			for i in range(numLines):
				# FEM solver
				exLine = self.exMat[i]
				f, JAC_i = self.solveOnce(exLine=exLine, triPerm=triPerm)

				# electrode
				diffArray = self.diffPairs(exLine, step, parser)
				vDiff = self.diff(f[self.elPos], diffArray)
				JAC_diff = self.diff(JAC_i, diffArray)

				fElement = f[self.elPos]
				b = self.smear(f, fElement, diffArray)

				v.append(vDiff)
				jacobian.append(JAC_diff)
				b_matrix.append(b)

		# update output
		pde_result = namedtuple("pde_result", output)
		hasil = pde_result(jac=np.vstack(jacobian), v=np.hstack(v), b_matrix=np.vstack(b_matrix))
		return hasil

	def solveJacMp(self, i):
		# FEM solver
		exLine = self.exMat[i]
		f, JAC_i = self.solveOnce(exLine=exLine, triPerm=self.triPerm)

		# electrode
		diffArray = self.diffPairs(exLine, self.step, self.parser)
		vDiff = self.diff(f[self.elPos], diffArray)
		JAC_diff = self.diff(JAC_i, diffArray)

		fElement = f[self.elPos]
		b = self.smear(f, fElement, diffArray)
		return [vDiff,JAC_diff,b]

	def solveOnce(self, exLine, triPerm):
		b = self.naturalBoundary(exLine=exLine)
		refElec = self.elPos[0]
		kGlobal, kElement = self.assemble(self.nodeXY, self.element, perm=triPerm, ref=refElec)
		# print(kGlobal.dtype)

		# electrode impedance
		rMatrix = la.inv(kGlobal) # scipy use here, lama
		# rMatrix = IM.inverse(kGlobal)

		# nodes potential
		f = np.dot(rMatrix, b).ravel()

		# pertubation on each element, Je = R*J*Ve
		ne = len(self.elPos)
		JAC = np.zeros((ne, self.elNum), dtype='complex')
		rEl = rMatrix[self.elPos]

		# bangkitkan matriks jacobian kolom by kolom, element wise
		for i in range(self.elNum):
			electroImpedance = self.element[i,:]
			JAC[:,i] = np.dot(np.dot(rEl[:,electroImpedance], kElement[i]), f[electroImpedance])

		return f, JAC

	def diffPairs(self, exLine, mStep, parser=None):
		drv_A = np.where(exLine==1)[0][0]
		drv_B = np.where(exLine==-1)[0][0]
		l = len(exLine)
		i0 = drv_A if parser is 'fmmu' else 0

		# build
		v = []
		for i in range(i0, i0+l):
			m = i%l
			n = (m+mStep)%l
			if not(m==drv_A or m==drv_B or n==drv_A or n==drv_B):
				v.append([n,m])

		return np.array(v)

	def diff(self, v, pairs):
		i = pairs[:,0]
		j = pairs[:,1]

		vDiff = v[i] - v[j]
		return vDiff

	def smear(self, f, fb, pairs):
		b_matrix = []
		for i,j in pairs:
			fMin, fMax = min(fb[i], fb[j]), max(fb[i], fb[j])
			b_matrix.append((fMin<f)&(f<=fMax))

		return np.array(b_matrix)

	def naturalBoundary(self, exLine):
		b = np.zeros((self.noNum, 1))
		aPos = self.elPos[np.where(exLine==1)]
		bPos = self.elPos[np.where(exLine==-1)]
		b[aPos] = 1.
		b[bPos] = -1.

		return b

	def assemble(self, nodeXY, element, perm=None, ref=0):
		noNum, _ = nodeXY.shape
		elNum, nVertices = element.shape

		if perm is None:
			perm = np.ones(elNum, dtype=np.float)

		if nVertices == 3:
			kLocal = self.kTriangle
		elif nVertices == 4:
			kLocal = self.kTetrahedron
		else:
			raise TypeError('dimensi error')

		kGlobal = np.zeros((noNum, noNum), dtype='complex')
		k_Element = np.zeros((elNum, nVertices, nVertices), dtype='complex')

		for ei in range(elNum):
			no = element[ei,:]
			xy = nodeXY[no,:]
			pe = perm[ei]

			# compute KIJ
			ke = kLocal(xy)
			k_Element[ei] = ke

			ij = np.ix_(no,no)

			kGlobal[ij] += (ke*pe)

		if 0<=ref < noNum:
			kGlobal[ref,:] = 0.
			kGlobal[:,ref] = 0.
			kGlobal[ref,ref] = 1.

		return kGlobal, k_Element

	def kTriangle(self, xy):
		s = xy[[2,0,1]] - xy[[1,2,0]]
		at = 0.5*la.det(s[[0,1]])

		kMatrix = np.dot(s,s.transpose())/(4.*at)
		return kMatrix

	def kTetrahedron(self, xy):
		s = xy[[2,3,0,1]] - xy[[1,2,3,0]]

		volume = 1./6 * la.det(s[[0,1,2]])

		ijPairs = [[0,1],[1,2],[2,3],[3,0]]
		signs = [1,-1,1,-1]
		a = [p*np.cross(s[i],s[j]) for(i,j),p in zip(ijPairs,signs)]
		a = np.array(a)

		# vector
		kMatrix = np.dot(a,a.transpose())/(36. * vt)
		return kMatrix
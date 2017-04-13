from __future__ import division, absolute_import, print_function

import numpy as np
from .EITbase import EITBase

class BackProjection(EITBase):
	def setup(self, weight='none'):
		self.params = {"weight": weight}

		if weight is 'simple':
			weight = self.simpleWeight(self.B.shape[0])
			self.H = weight * self.B

	def solve(self, v1, v0, normalize=True):
		if normalize:
			vn = -(v1 - v0)/np.sign(v0)
		else:
			vn = (v1 - v0)

		hasil = np.dot(self.H.transpose(), vn)
		return np.real(hasil)

	def solveGramSchmidt(self, v1, v0): # https://en.wikipedia.org/wiki/Gram–Schmidt_process
		a = np.dot(v1, v0)/np.dot(v0, v0)
		vn = -(v1 - a*v0)/np.sign(v0)
		hasil = np.dot(self.H.transpose(), vn)
		return np.real(hasil)

	def simpleWeight(self, numVoltages):
		d = np.sqrt(np.sum(self.node**2, axis=1))
		r = np.max(d)
		w = (1.01*r - d)/(1.01*r)

		weights = np.dot(np.ones((numVoltages, 1)), w.reshape(1,-1))
		return weights
from __future__ import division, absolute_import, print_function

import numpy as np
from .base import EITBase

class BackProjection(EITBase):
	def setup(self, weight='none'):
		self.params = {"weight": weight}

		if weight is 'simple':
			weight = self.simpleWeight(self.B.shape[0])
			self.H = weight * self.B

	def solve(self, v1, v0=None, normalize=True):
		if v0 is None:
			v0 = self.v0

		if normalize:
			vn = -(v1 - v0)/np.sign(self.v0)
		else:
			vn = (v1 - v0)

		hasil = np.dot(self.H.transpose(), vn)
		return np.real(hasil)

	def simpleWeight(self, numVoltages):
		d = np.sqrt(np.sum(self.nodeXY**2, axis=1))
		r = np.max(d)
		w = (1.01*r - d)/(1.01*r)

		weights = np.dot(np.ones(numVoltages, 1), w.reshape(1,-1))
		return weights
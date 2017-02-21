from __future__ import division, absolute_import, print_function

import numpy as np

from .FEM import Forward
from .utils import EIT_scanLines

class EITBase(object):
	def __init__(self, mesh, elPos, exMat=None, step=1, perm=1, parser='std'):
		self.mesh = mesh
		self.nodeXY = mesh['node']
		self.element = mesh['element']

		self.elPos = elPos
		self.parser = parser
		self.noNum, self.dim = self.nodeXY.shape
		self.elNum, self.nVertices = self.element.shape

		if exMat is None:
			exMat = EIT_scanLines(len(elPos), 1)

		if perm is None:
			perm = np.ones_like(mesh['alpha'])

		if np.size(perm)==1:
			self.perm = perm * np.ones(self.elNum)
		else:
			self.perm = perm
			
		self.exMat = exMat
		self.step = step

		# butuh forward problem
		forward = Forward(mesh, elPos)
		self.forward = forward
		# solve Jacobian using uniform sigma distribution
		res = forward.solve(exMat, step=step, perm=self.perm, parser=self.parser)
		self.Jacobian, self.v0, self.B = res.jac, res.v, res.b_matrix

		self.H = self.B

		self.params = {}
		self.setup()
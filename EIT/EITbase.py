from __future__ import division, absolute_import, print_function

import numpy as np

class EITBase(object):
	def __init__(self, mesh, forward):
		self.node = mesh['node']
		self.element = mesh['element']
		self.alpha = mesh['alpha']
		self.jacobian = forward.jac
		self.B = forward.b_matrix
		self.H = self.B

		self.params = {}
		self.setup()
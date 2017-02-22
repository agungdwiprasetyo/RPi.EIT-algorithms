from __future__ import division, absolute_import, print_function

import numpy as np

class EITBase(object):
	def __init__(self, node, boolMatrix):
		self.node = node
		self.B = boolMatrix
		self.H = self.B

		self.params = {}
		self.setup()
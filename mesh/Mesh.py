from __future__ import division, absolute_import, print_function

import numpy as np
from .distmesh import build
from .mesh_circle import MeshCircle
from .utils import check_order
from .shape import unit_circle, unit_ball, area_uniform
from .shape import fix_points_fd, fix_points_ball

class Mesh(object):
	def __init__(self, n_Elec, fd=None, fh=None, pFix=None, bbox=None, h0=0.1):
		if bbox is None:
			bbox = [[-1, -1], [1, 1]]
		bbox = np.array(bbox)
		dimensi = bbox.shape[1]
		if dimensi not in [2,3]:
			raise TypeError('Mesh hanya mendukung objek 2D atau 3D')
		if bbox.shape[0]!=2:
			raise TypeError('init lower and upper bound')

		# set tipe mesh
		if dimensi==2:
			if fd is None:
				fd = unit_circle
			if pFix is None:
				pFix = fix_points_fd(fd, n_el=n_Elec)
		elif dimensi==3:
			if fd is None:
				fd = unit_ball
			if pFix is None:
				pFix = fix_points_ball(n_el=n_Elec)

		if fh is None:
			fh = area_uniform

		self.node, self.element = build(fd, fh, pfix=pFix, bbox=bbox, h0=h0)
		self.element = check_order(self.node, self.element)
		self.elPos = np.arange(n_Elec)
		self.alpha = np.ones(self.element.shape[0], dtype=np.float)

	def getMesh(self):
		mesh = {'element': self.element, 'node': self.node, 'alpha': self.alpha}
		return mesh

	def getElectrode(self):
		return self.elPos

	def setAlpha(self, anomaly=None, background=None):
		triCenters = np.mean(self.node[self.element], axis=1)
		alpha = self.alpha
		n = np.size(self.alpha)

		if background is not None:
			alpha = background*np.ones(n, dtype='complex')

		if anomaly is not None:
			for _, attr in enumerate(anomaly):
				d = attr['d']
				if 'z' in attr:
					index = np.sqrt((triCenters[:,0]-attr['x'])**2 + (triCenters[:,1]-attr['y'])**2 + (triCenters[:,2]-attr['z'])**2)<d
				else:
					index = np.sqrt((triCenters[:,0]-attr['x'])**2 + (triCenters[:,1]-attr['y'])**2)<d

				alpha[index] = attr['alpha']

		newMesh = {'element': self.element, 'node': self.node, 'alpha': alpha}
		return alpha
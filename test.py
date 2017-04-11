#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from EIT.BackProjection import BackProjection

import time

nElectrode = 16
sizeImage = (13,6)
axisSize = [-1.2, 1.2, -1.2, 1.2]

createMesh = Mesh(nElectrode, h0=0.05)
mesh = createMesh.getMesh()
elPos = createMesh.getElectrode()

nodeXY = mesh['node']
element = mesh['element']
alpha0 = createMesh.setAlpha(background=7.5)
# alpha = permitivitas objek/konduktivitas, makin besar melebihi nilai background makin berwarna merah, artinya nilai impedansi makin kecil
anomaly = [{'x': 0.46, 'y': 0.5, 'd': 0.2, 'alpha': 3},
           {'x': 0, 'y': -0.3, 'd': 0.1, 'alpha': 3},
           {'x': 0, 'y': 0, 'd': 0.1, 'alpha': 15}]
alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=7.5)

deltaAlpha = np.real(alpha1 - alpha0)

step = 1

exMat = EIT_scanLines(nElectrode)
start = time.time()
forward = Forward(mesh, elPos)
f0 = forward.solve(exMat, step=step, perm=alpha0)
fin = time.time()
print("Waktu Forward Problem Solver = %.4f" %(fin-start))
f1 = forward.solve(exMat, step=step, perm=alpha1)

inverse = BackProjection(mesh=mesh, forward=f0)
resInverse = inverse.solveGramSchmidt(f1.v, f0.v)

# plot mesh
fig = plt.figure(figsize=sizeImage)
gs = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, deltaAlpha)
ax1.set_title(r'(a) $\Delta$ Conductivity')
ax1.axis(axisSize)
ax1.axis('off')

# plot citra akhir
ax2 = fig.add_subplot(gs[0, 1])
ax2.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, resInverse)
ax2.set_title(r'(b) Back Projection')
ax2.axis(axisSize)
ax2.axis('off')

plt.show()
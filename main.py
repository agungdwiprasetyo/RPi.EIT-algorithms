#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
import EIT.backProjection as BP

import time


sizeImage = (8,8)
axisSize = [-1.2, 1.2, -1.2, 1.2]
jumlahElektroda = 16

initMesh = Mesh(jumlahElektroda, h0=0.1)
mesh = initMesh.getMesh()
elPos = initMesh.getElectrode()

nodeXY = mesh['node']
element = mesh['element']

anomaly = [{'x': 0.4,  'y': 0,    'd': 0.1, 'alpha': 10},
           {'x': -0.4, 'y': 0,    'd': 0.1, 'alpha': 10},
           {'x': 0,    'y': 0.5,  'd': 0.1, 'alpha': 0.1},
           {'x': 0,    'y': -0.5, 'd': 0.1, 'alpha': 0.1}]
mesh0 = initMesh.setAlpha(background=1.0)
mesh1 = initMesh.setAlpha(anomaly=anomaly, background=1.0)
alpha = np.real(mesh1['alpha'] - mesh0['alpha'])

step = 1
exMat = EIT_scanLines(jumlahElektroda, 1)

start = time.time()
forward = Forward(mesh, elPos)
f0 = forward.solve(exMat, step=step, perm=mesh0['alpha'])
f1 = forward.solve(exMat, step=step, perm=mesh1['alpha'])
fin = time.time()
print("waktu forward problem solver = %.4f" %(fin-start))

# inverse problem with BP
start = time.time()
inverse = BP.BackProjection(mesh, elPos, exMat, step=1, parser='std')
hasilBP = inverse.solve(f1.v, f0.v, normalize=True)
fin = time.time()
print("waktu inverse problem solver = %.4f" %(fin-start))

# plot mesh
fig = plt.figure(figsize=sizeImage)
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, alpha)
ax1.set_title(r'(a) $\Delta$ Conductivity')
ax1.axis(axisSize)
ax1.axis('off')

# plot citra akhir
ax2 = fig.add_subplot(gs[0, 1])
ax2.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, hasilBP)
ax2.set_title(r'(b) Back Projection')
ax2.axis(axisSize)
ax2.axis('off')

plt.show()
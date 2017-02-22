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


sizeImage = (13,6)
axisSize = [-1.2, 1.2, -1.2, 1.2]
jumlahElektroda = 16
arusInjeksi = 10

initMesh = Mesh(jumlahElektroda, h0=0.08)
mesh = initMesh.getMesh()
elPos = initMesh.getElectrode()

# set simulation model
nodeXY = mesh['node']
element = mesh['element']
alpha0 = initMesh.setAlpha(background=arusInjeksi)

anomaly = [{'x': -0.625, 'y': 0.25, 'd': 0.3, 'alpha': 1},
           {'x': 0.46, 'y': 0.5, 'd': 0.3, 'alpha': 1},
           {'x': -0.4, 'y': -0.5, 'd': 0.1, 'alpha': 20},
           {'x': 0.6, 'y': -0.3, 'd': 0.1, 'alpha': 20}]
alpha1 = initMesh.setAlpha(anomaly=anomaly ,background=arusInjeksi) 

deltaAlpha = np.real(alpha1 - alpha0)

step = 1
exMat = EIT_scanLines(jumlahElektroda, 1)

# solve Forward Model
start = time.time()
forward = Forward(mesh, elPos)
f0 = forward.solve(exMat, step=step, perm=alpha0)
# f1 = forward.solve(exMat, step=step, perm=alpha1)
refFEM = f0.v
boolMatrix = f0.b_matrix
print(boolMatrix)
fin = time.time()
print("Waktu Forward Problem Solver = %.4f" %(fin-start))

# impor data from EIT instrument
data = np.loadtxt("data/dataukur.txt")
ref = np.loadtxt("data/dataref.txt")

# solve inverse problem with BP
start = time.time()
inverse = BackProjection(nodeXY, boolMatrix)
hasilBP = inverse.solve(data, refFEM)
fin = time.time()
print("Waktu Inverse Problem Solver = %.4f" %(fin-start))

# plot mesh
fig = plt.figure(figsize=sizeImage)
gs = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, deltaAlpha, shading='flat')
ax1.plot(nodeXY[elPos, 0], nodeXY[elPos, 1], 'ro')
ax1.set_title(r'(a) $\Delta$ Conductivity')
ax1.axis(axisSize)
ax1.axis('off')

# plot image
ax2 = fig.add_subplot(gs[0, 1])
ax2.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, hasilBP)
ax2.set_title(r'(b) Back Projection')
ax2.axis(axisSize)
ax2.axis('off')

plt.show()
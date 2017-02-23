#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from EIT.BackProjection import BackProjection
from EIT.Jacobian import Jacobian
from EIT.GREIT import GREIT

import time


sizeImage = (12,12)
axisSize = [-1.2, 1.2, -1.2, 1.2]
jumlahElektroda = 16
arusInjeksi = 7.5 # miliAmpere
kerapatan = 0.05

createMesh = Mesh(jumlahElektroda, h0=kerapatan)
mesh = createMesh.getMesh()
elPos = createMesh.getElectrode()

# set simulation model
nodeXY = mesh['node']
element = mesh['element']
alpha0 = createMesh.setAlpha(background=arusInjeksi)

anomaly = [{'x': -0.625, 'y': 0.25, 'd': 0.3, 'alpha': 1},
           {'x': 0.46, 'y': 0.5, 'd': 0.3, 'alpha': 1},
           {'x': -0.4, 'y': -0.5, 'd': 0.1, 'alpha': 20},
           {'x': 0.6, 'y': -0.3, 'd': 0.1, 'alpha': 20}]
alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=arusInjeksi) 

deltaAlpha = np.real(alpha1 - alpha0)

step = 1
exMat = EIT_scanLines(jumlahElektroda)

# solve Forward Model
start = time.time()
forward = Forward(mesh, elPos)
f0 = forward.solve(exMat, step=step, perm=alpha0)
fin = time.time()
print("Waktu Forward Problem Solver = %.4f" %(fin-start))

# impor data from EIT instrument, and FEM result
data = np.loadtxt("data/dataukur.txt")
ref = np.loadtxt("data/dataref.txt")
refFEM = f0.v

# ----------------------------------------- solve inverse problem with BP -------------------------------------------
start = time.time()
inverseBP = BackProjection(mesh=mesh, forward=f0)
hasilBP = inverseBP.solveGramSchmidt(data, refFEM)
fin = time.time()
print("Waktu Inverse Problem Solver (BP)  = %.4f" %(fin-start))

# ------------------------------------------ solve inverse problem with Jacobian ------------------------------------
start = time.time()
inverseJAC = Jacobian(mesh=mesh, forward=f0)
hasilJAC = inverseJAC.solve(data, refFEM)
fin = time.time()
print("Waktu Inverse Problem Solver (JAC) = %.4f" %(fin-start))

# ------------------------------------------- solve inverse problem with GREIT -------------------------------------
start = time.time()
inverseGREIT = GREIT(mesh=mesh, forward=f0)
ds = inverseGREIT.solve(data, refFEM)
x, y, hasilGREIT = inverseGREIT.mask_value(ds, mask_value=np.NAN)
fin = time.time()
print("Waktu Inverse Problem Solver (GREIT) = %.4f" %(fin-start))
# ------------------------------------------------- END inverse problem --------------------------------------------

# plot mesh
fig = plt.figure(figsize=sizeImage)
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, deltaAlpha, shading='flat')
ax1.plot(nodeXY[elPos, 0], nodeXY[elPos, 1], 'ro')
ax1.set_title(r'(a) $\Delta$ Finite Element Method')
ax1.axis(axisSize)
ax1.axis('off')

# plot image from BP
ax2 = fig.add_subplot(gs[0, 1])
ax2.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, hasilBP)
ax2.set_title(r'(b) Back Projection')
ax2.axis(axisSize)
ax2.axis('off')

# plot image from Jacobian
ax3 = fig.add_subplot(gs[1, 0])
ax3.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, np.real(hasilJAC))
ax3.set_title(r'(c) Jacobian')
ax3.axis(axisSize)
ax3.axis('off')

# plot image from GREIT
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(np.real(hasilGREIT), interpolation='nearest')
ax4.set_title(r'(d) GREIT')
ax4.axis('off')

plt.show()
#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from solver.InverseSolver import InverseSolver
import matplotlib.pyplot as plt

import time

nElectrode = 16
sizeImage = (13,6)
axisSize = [-1.2, 1.2, -1.2, 1.2]

print("Pemodelan objek background...")
createMesh = Mesh(nElectrode, h0=0.07)
mesh = createMesh.getMesh()
elPos = createMesh.getElectrode()

nodeXY = mesh['node']
element = mesh['element']
alpha0 = createMesh.setAlpha(background=1.0)
# alpha = permitivitas objek/konduktivitas, makin besar melebihi nilai background makin berwarna merah, artinya nilai impedansi makin kecil
print("Membuat manual contoh objek...")
anomaly = [{'x': 0.46, 'y': 0.5, 'd': 0.2, 'alpha': 2.5},
		   {'x': -0.46, 'y': -0.5, 'd': 0.2, 'alpha': 1.5}]
alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=1.0)
deltaAlpha = np.real(alpha1 - alpha0)

# draw konduktivitas
print("Plot konduktivitas contoh model...")
fig, ax = plt.subplots()
im = ax.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, deltaAlpha, shading='flat', cmap=plt.cm.viridis)
ax.set_title(r'$\Delta$ Conductivities')
fig.colorbar(im)
ax.axis('equal')
fig.set_size_inches(6, 4)
plt.show()

step = 1
exMat = EIT_scanLines(nElectrode)
start = time.time()
forward = Forward(mesh, elPos)
print("Penyelesaian forward problem untuk model background...")
resFEM1 = forward.solve(exMat, step=step, perm=alpha0)
fin = time.time()
# print("Waktu Forward Problem Solver = %.4f" %(fin-start))
print("Penyelesaian forward problem untuk contoh model...")
resFEM2 = forward.solve(exMat, step=step, perm=alpha1)
dataA = np.loadtxt("../RPi.EIT-web/dataObjek/PhantomAManualNew.txt")
dataB = np.loadtxt("../RPi.EIT-web/dataObjek/PhantomBUkurManualNew.txt")
dataC = np.loadtxt("../RPi.EIT-web/dataObjek/PhantomCSimulasiNew.txt")

# Algor--> BP=Back Projection, JAC=Jacobian, GREIT=Graz
start2 = time.time()
print("Penyelesaian inverse problem, menyimpan matriks jacobian dan distribusi potensial...")
inverse = InverseSolver(mesh=mesh, forward=resFEM1)
while True:
	z = input("1. Data contoh model\n"+
		"2. Data phantom A\n"+
		"3. Data phantom B\n"+
		"4. Data phantom C\n"+
		"Pilih data: ")
	if z=="1":
		objek = resFEM2.v
	elif z=="2":
		objek=dataA
	elif z=="3":
		objek=dataB
	elif z=="4":
		objek=dataC
	else:
		break
	inverse.solve(data=objek)
	# print("Waktu inverse Problem Solver = %.4f" %(time.time()-start2))
	inverse.plot(size=[-1, 1, -1, 1], colorbar=True, showPlot=True)
	print("\n")
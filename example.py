#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from solver.InverseSolver import InverseSolver
import matplotlib.pyplot as plt

import time

# permitivitas objek/konduktivitas, makin besar melebihi nilai background makin berwarna merah, artinya nilai impedansi makin kecil

nElectrode = 16
sizeImage = (13,6)
axisSize = [-1.2, 1.2, -1.2, 1.2]

# bangkitkan model 2D dari mesh
print("Pemodelan 2D mesh...")
createMesh = Mesh(nElectrode, h0=0.07)
mesh = createMesh.getMesh()
elPos = createMesh.getElectrode()
nodeXY = mesh['node']
element = mesh['element']

# inisialisasi background untuk model homogen
alpha0 = createMesh.setAlpha(background=1.0)

# membuat contoh model objek dengan set anomaly secara manual
print("Membuat manual contoh objek dengan set anomaly sesuai koordinat yang diinginkan...")
anomaly = [{'x': 0.46, 'y': 0.5, 'd': 0.2, 'alpha': 2.5},
		   {'x': -0.46, 'y': -0.5, 'd': 0.2, 'alpha': 1.5}]
alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=1.0)
deltaAlpha = np.real(alpha1 - alpha0)

# show fem: tampilkan finite element model
print("Plot konduktivitas contoh model...")
fig, ax = plt.subplots()
im = ax.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, deltaAlpha, edgecolors='k', shading='flat', alpha=0.5, cmap=plt.cm.Greys)
ax.set_title(r'$\Delta$ Conductivities')
fig.colorbar(im)
ax.axis('equal')
fig.set_size_inches(6, 4)
plt.show()

exMat = EIT_scanLines(nElectrode)

# inisialisasi forward problem
forward = Forward(mesh, elPos)

# solve forward problem untuk model homogen
print("Penyelesaian forward problem untuk model homogen...")
resFEM1 = forward.solve(exMat, step=1, perm=alpha0)

# solve forward problem untuk contoh model, dari sini dihasilkan data sintetis dari contoh model objek yang diset anomalinya
print("Penyelesaian forward problem untuk contoh model...")
resFEM2 = forward.solve(exMat, step=1, perm=alpha1)

# load contoh data sintetis
dataA = np.loadtxt("./data/PhantomA.txt")
dataB = np.loadtxt("./data/PhantomB.txt")
dataC = np.loadtxt("./data/PhantomC.txt")

# Inverse Problem. Algor--> BP=Back Projection, JAC=Gauss-Newton, GREIT=Graz
# Create Inverse Model dari model homogen, 
print("Penyelesaian inverse problem, menyimpan matriks jacobian dan distribusi potensial...")
inverse = InverseSolver(mesh=mesh, forward=resFEM1)
while True:

	# pilih data sintetis yang akan dicitrakan
	z = input("1. Data contoh model\n"+
		"2. Data phantom A\n"+
		"3. Data phantom B\n"+
		"4. Data phantom C\n"+
		"Pilih data: ")

	if z=="1":
		objek = resFEM2.v
	elif z=="2":
		objek = dataA
	elif z=="3":
		objek = dataB
	elif z=="4":
		objek = dataC
	else:
		break

	# solve inverse problem untuk data sintetis yang dipilih
	inverse.solve(data=objek, algor="BP")

	# plot hasil citra dari inverse model
	inverse.plot(size=[-1, 1, -1, 1], colorbar=True, showPlot=True)
	print("\n")
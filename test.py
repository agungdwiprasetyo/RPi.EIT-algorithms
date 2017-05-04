#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function
import numpy as np

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from solver.InverseSolver import InverseSolver

import time

nElectrode = 16
sizeImage = (13,6)
axisSize = [-1.2, 1.2, -1.2, 1.2]

createMesh = Mesh(nElectrode, h0=0.08)
mesh = createMesh.getMesh()
elPos = createMesh.getElectrode()

nodeXY = mesh['node']
element = mesh['element']
alpha0 = createMesh.setAlpha(background=7.5)
# alpha = permitivitas objek/konduktivitas, makin besar melebihi nilai background makin berwarna merah, artinya nilai impedansi makin kecil
anomaly = [{'x': 0.46, 'y': 0.5, 'd': 0.2, 'alpha': 10},{'x': -0.46, 'y': -0.5, 'd': 0.2, 'alpha': 9}]
alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=7.5)

deltaAlpha = np.real(alpha1 - alpha0)

step = 1

exMat = EIT_scanLines(nElectrode)
start = time.time()
forward = Forward(mesh, elPos)
f0 = forward.solve(exMat, step=step, perm=alpha0)
fin = time.time()
print("Waktu Forward Problem Solver = %.4f" %(fin-start))
# f1 = forward.solve(exMat, step=step, perm=alpha1)
print(f0.jac.shape)
data = np.loadtxt("../RPi.EIT-web/dataObjek/DataPenelitianTumit.txt")

# Algor--> BP=Back Projection, JAC=Jacobian, GREIT=Graz
start2 = time.time()
inverse = InverseSolver(mesh=mesh, forward=f0)
inverse.solve(algor="BP", data=data)
print("Waktu inverse Problem Solver = %.4f" %(time.time()-start2))
inverse.plot(size=[-1, 1, -1, 1], colorbar=False, showPlot=True)
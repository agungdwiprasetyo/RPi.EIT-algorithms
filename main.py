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

from socketIO_client import SocketIO, LoggingNamespace
from API.API import API

import time

host = 'localhost'
port = 3456

socketIO = SocketIO(host, port, LoggingNamespace)
socketIO.emit('raspiConnect', {'status': True})
api = API()

def run(*args):
	responseData = args[0]
	print(responseData)
	sizeImage = (12,12)
	axisSize = [-1.2, 1.2, -1.2, 1.2]
	jumlahElektroda = 16
	arusInjeksi = responseData['arus'] #default 7.5 miliAmpere
	kerapatan = responseData['kerapatan']
	dataVolt = responseData['data']
	algor = responseData['algor']
	datetime = (time.strftime("%Y%m%d-") + time.strftime("%H%M%S"))
	waktu = 0
	direktori = "../rpieit-web/img/results/"
	filename = str(datetime)+'-'+algor+'.png'

	createMesh = Mesh(jumlahElektroda, h0=kerapatan)
	mesh = createMesh.getMesh()
	elPos = createMesh.getElectrode()

	# set simulation model
	nodeXY = mesh['node']
	element = mesh['element']
	alpha0 = createMesh.setAlpha(background=arusInjeksi)

	anomaly = [{'x': 1.3, 'y': 0.5, 'd': 0.3, 'alpha': 1},
	           {'x': 0.46, 'y': 0.5, 'd': 0.3, 'alpha': 20}]
	alpha1 = createMesh.setAlpha(anomaly=anomaly ,background=arusInjeksi) 

	deltaAlpha = np.real(alpha1 - alpha0)

	step = 1
	exMat = EIT_scanLines(jumlahElektroda)

	# solve Forward Model
	print("Starting reconstruction with "+algor+"...")
	start = time.time()
	forward = Forward(mesh, elPos)
	f0 = forward.solve(exMat, step=step, perm=alpha0)
	fin = time.time()
	waktu+=(fin-start)
	print("Waktu Forward Problem Solver = %.4f" %(fin-start))

	# impor data from EIT instrument, and result from FEM
	data = np.loadtxt("data/"+dataVolt+".txt")
	ref = f0.v

	# ----------------------------------------- solve inverse problem with BP -------------------------------------------
	if(algor=="BP"):
		start = time.time()
		inverseBP = BackProjection(mesh=mesh, forward=f0)
		resInverse = inverseBP.solveGramSchmidt(data, ref)
		fin = time.time()
		waktu+=(fin-start)
		print("Waktu Inverse Problem Solver (BP)  = %.4f" %(fin-start))

	# ------------------------------------------ solve inverse problem with Jacobian ------------------------------------
	elif(algor=="JAC"):
		start = time.time()
		inverseJAC = Jacobian(mesh=mesh, forward=f0)
		hasilJAC = inverseJAC.solve(data, ref)
		resInverse = np.real(hasilJAC)
		fin = time.time()
		waktu+=(fin-start)
		print("Waktu Inverse Problem Solver (JAC) = %.4f" %(fin-start))

	# ------------------------------------------- solve inverse problem with GREIT -------------------------------------
	elif(algor=="GREIT"):
		start = time.time()
		inverseGREIT = GREIT(mesh=mesh, forward=f0)
		ds = inverseGREIT.solve(data, ref)
		x, y, hasilGREIT = inverseGREIT.mask_value(ds, mask_value=np.NAN)
		resInverse = np.real(hasilGREIT)
		fin = time.time()
		waktu+=(fin-start)
		print("Waktu Inverse Problem Solver (GREIT) = %.4f" %(fin-start))
	# ------------------------------------------------- END inverse problem --------------------------------------------

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	if(algor=="GREIT"):
		im = ax1.imshow(resInverse, interpolation='nearest')
		# fig.colorbar(im)
	else:
		ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, resInverse)
	ax1.axis('equal')
	plt.axis([-1, 1, -1, 1])
	fig.set_size_inches(6, 6)
	plt.axis('off')
	fig.savefig(direktori+filename, dpi=300)
	plt.show()

	# # plot image from GREIT
	# ax4 = fig.add_subplot(gs[1, 1])
	# ax4.imshow(np.real(hasilGREIT), interpolation='nearest')
	# ax4.set_title(r'(d) GREIT')
	# ax4.axis('off')

	print('Finish')
	socketIO.emit('status', {'sukses': True, 'waktu': int(waktu)})
	api.postImage(filename, kerapatan, arusInjeksi, algor, dataVolt)

	plt.show()


# main function
if __name__ == '__main__':
	print ("Listening server...")
	socketIO.on('startReconstruction', run)
	socketIO.wait()
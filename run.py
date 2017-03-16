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
from config import * # import main variable

import time


socketIO = SocketIO(host, port, LoggingNamespace)
api = API(host, port)

def run(*args):
	responseData = args[0]
	print(responseData)

	axisSize = [-1, 1, -1, 1]
	jumlahElektroda = 16
	arusInjeksi = responseData['arus'] #default 7.5 miliAmpere
	kerapatan = responseData['kerapatan']
	iddata = responseData['iddata']
	dataVolt = str(responseData['data'])
	algor = str(responseData['algor'])
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

	# ----------------------------------------- solve Forward Model with FEM -------------------------------------------
	print("Starting reconstruction with "+algor+"...")
	start = time.time()
	forward = Forward(mesh, elPos)
	f0 = forward.solve(exMat, step=step, perm=alpha0)
	fin = time.time()
	waktu+=(fin-start)
	print("Waktu Forward Problem Solver = %.4f" %(fin-start))

	# impor data from EIT instrument, and result from FEM
	data = np.loadtxt("data/"+dataVolt)
	ref = f0.v

	# ----------------------------------------- solve inverse problem with BP -------------------------------------------
	if(algor=="BP"):
		start = time.time()
		inverseBP = BackProjection(mesh=mesh, forward=f0)
		resInverse = inverseBP.solveGramSchmidt(data, ref)
		fin = time.time()
		print("Waktu Inverse Problem Solver (BP)  = %.4f" %(fin-start))

	# ------------------------------------------ solve inverse problem with Jacobian ------------------------------------
	elif(algor=="JAC"):
		start = time.time()
		inverseJAC = Jacobian(mesh=mesh, forward=f0)
		hasilJAC = inverseJAC.solve(data, ref)
		resInverse = np.real(hasilJAC)
		fin = time.time()
		print("Waktu Inverse Problem Solver (JAC) = %.4f" %(fin-start))

	# ------------------------------------------- solve inverse problem with GREIT -------------------------------------
	elif(algor=="GREIT"):
		start = time.time()
		inverseGREIT = GREIT(mesh=mesh, forward=f0)
		ds = inverseGREIT.solve(data, ref)
		x, y, hasilGREIT = inverseGREIT.mask_value(ds, mask_value=np.NAN)
		resInverse = np.real(hasilGREIT)
		fin = time.time()
		print("Waktu Inverse Problem Solver (GREIT) = %.4f" %(fin-start))

	elif(algor=="ART"):
		print("Algebraic")
	# ------------------------------------------------- END inverse problem --------------------------------------------

	waktu+=(fin-start)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	if(algor=="GREIT"):
		im = ax1.imshow(resInverse, interpolation='none')
		ax1.axis([-1, 1, -1, 1])
		fig.set_size_inches(6,6)
		plt.axis('off')
		ax1.axis('equal')
	else:
		im = ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], element, resInverse)
		ax1.axis('equal')
		ax1.axis([-1, 1, -1, 1])
		fig.set_size_inches(6,6)
		plt.axis('off')
		
	if(responseData['colorbar']):
		fig.colorbar(im)

	print('Finish')
	fig.savefig(direktori+filename, dpi=300)
	api.postImage(filename, kerapatan, algor, iddata)
	socketIO.emit('finishReconstruction', {'sukses': True, 'waktu': int(waktu), 'filename': filename})
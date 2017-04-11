from __future__ import division, absolute_import, print_function
import numpy as np

from mesh.Mesh import Mesh
from EIT.FEM import Forward
from EIT.utils import EIT_scanLines
from solver.InverseSolver import InverseSolver

from socketIO_client import SocketIO, LoggingNamespace
from API.API import API
from config import * # import main variable

import time

socketIO = SocketIO(host, port, LoggingNamespace)
api = API(host, port)

def run(*args):
	responseData = args[0]
	print(responseData)

	# generate parameter
	axisSize = [-1, 1, -1, 1]
	jumlahElektroda = 16
	tipe = str(responseData['tipe'])
	arusInjeksi = responseData['arus'] #default 7.5 miliAmpere
	kerapatan = responseData['kerapatan']
	iddata = responseData['iddata']
	dataVolt = responseData['data']
	algor = str(responseData['algor'])
	datetime = (time.strftime("%Y%m%d-") + time.strftime("%H%M%S"))
	waktu = time.time()
	direktori = "./RPi.EIT-web/img/results/"
	filename = str(datetime)+'-'+algor+'.png'

	# set simulation model
	createMesh = Mesh(jumlahElektroda, h0=kerapatan)
	mesh = createMesh.getMesh()
	elPos = createMesh.getElectrode()
	alpha0 = createMesh.setAlpha(background=arusInjeksi)
	step = 1
	exMat = EIT_scanLines(jumlahElektroda)

	# impor data from EIT instrument or file
	if(tipe == "fromraspi"):
		data = np.hstack(dataVolt)
	else:
		data = np.loadtxt("./RPi.EIT-web/dataObjek/"+str(dataVolt))

	# Solve model with forward problem (FEM)
	forward = Forward(mesh, elPos)
	f0 = forward.solve(exMat=exMat, step=step, perm=alpha0)

	# Solve inverse problem
	inv = InverseSolver(mesh=mesh, forward=f0)
	inv.solve(algor=algor, data=data)

	# plot
	fig = inv.plot(size=axisSize,colorbar=responseData['colorbar'])
	fig.savefig(direktori+filename, dpi=300)
	waktu = time.time()-waktu

	# report to server
	api.postImage(filename, kerapatan, algor, iddata)
	socketIO.emit('finishReconstruction', {'sukses': True, 'waktu': int(waktu), 'filename': filename})
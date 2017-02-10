#!/usr/bin/env python3

import mesh
from mesh.quality import fstats # buat ngecek jumlah node dan element model citra
from eit.utils import eit_scan_lines
from eit.fem import forward
import eit.jac as jac # Gauss-Newton Solver
import eit.bp as bp # Back-Projection
import numpy as np
import matplotlib.pyplot as plot
from random import uniform, randint

def startSolve(forwardProblem, exLine, alphaBaru):
	# mulai proses forward problem
	ff,jacobi = forwardProblem.solve_once(exLine,alphaBaru) # ff masih terdapat bilangan imajiner
	ff = np.real(ff) # ff di-real kan
	# print(min(ff))
	vf = np.linspace(min(ff), max(ff), 32) # membagi array ff dari nilai minimal ke maksimal menjadi sebanyak 32 dengan selisih tiap2 nilai sama
	# print(vf)
	# plot
	fig = plot.figure("Hasil Plot Forward Problem")
	ax1 = fig.add_subplot(111)
	ax1.tricontour(nodeXY[:, 0], nodeXY[:, 1], numElemen, ff, vf, linewidth=0.5, cmap=plot.cm.viridis)
	ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], numElemen, np.real(alphaBaru), edgecolors='k', shading='flat', alpha=0.5, cmap=plot.cm.Greys)
	ax1.plot(nodeXY[elektroda, 0], nodeXY[elektroda, 1], 'ro')
	ax1.set_title('Forward Problem')
	ax1.axis('equal')
	fig.set_size_inches(6, 4)
	plot.show()

	# inverseProblemSolver()

def inverseProblemSolver(forwardProblem, inputVol, mesh, alphaBaru):
	nodeXY = mesh['node']
	numElemen = mesh['element']
	alpha = mesh['alpha']

	print("Sedang memproses inverse problem...")
	# proses inverse problem
	f0 = forwardProblem.solve(inputVol, step=1, perm=alpha)
	f1 = forwardProblem.solve(inputVol, step=1, perm=alphaBaru)
	print(f0)
	# eit = jac.JAC(mesh, elektroda, exMtx=inputVol, step=1, perm=1., parser='std',p=0.30, lamb=1e-4, method='kotre')	# Gauss-Newton Solver
	eit = bp.BP(mesh, elektroda, exMtx=inputVol, step=1, parser='std', weight='none') # Back-Projection
	ds = eit.solve(f1.v, f0.v)
	print("Sukses.")
	# plot
	fig = plot.figure("Hasil Plot Inverse Problem")
	ax1 = fig.add_subplot(111)
	plot.tripcolor(nodeXY[:, 0], nodeXY[:, 1], numElemen, np.real(ds),shading='flat', cmap=plot.cm.viridis)
	plot.colorbar()
	ax1.set_title('Inverse Problem')
	plot.axis('equal')
	fig.set_size_inches(6, 4)
	plot.show()


# MAIN FUNCTION
if __name__ == "__main__":
	jumlahElektroda = 16

	msh, elektroda = mesh.create(jumlahElektroda, h0=0.1) # init mesh
	print (elektroda)

	nodeXY = msh['node']
	numElemen = msh['element']
	alpha = msh['alpha']

	fstats(nodeXY, numElemen) # print jumlah node dan element

	Mat1 = eit_scan_lines(jumlahElektroda, 7) # membuat matriks diagonal yg nilainya 1 yg ukurannya sebesar 16*16
	exLine = Mat1[0].ravel()
	print("Mat1 = ",Mat1,"\n")
	FP = forward(msh, elektroda) # init forward problem

	mat2 = [1, 2, 3, 4, 5, 6, 7, 8]

	# Buat input tegangan acak
	inputVol = np.zeros((jumlahElektroda, jumlahElektroda))
	for i in range(jumlahElektroda):
		# inputVol[i, i % jumlahElektroda] = randint(-1,1)
		inputVol[i, i % jumlahElektroda] = 1
		inputVol[i, (i+1) % jumlahElektroda] = -1
	print (inputVol)

	# Memulai proses pencitraan
	x = 0.2
	y = 0.5
	d = 0.3
	alp = 3
	# ngeset anomali
	anomaly = [{'x': x, 'y': y, 'd': d, 'alpha': alp},
				{'x': -x, 'y': -y, 'd': d, 'alpha': alp}] # x dan y menyatakan posisi tengah dari anomali yg terbentuk, d menyatakan range
	ms_baru = mesh.set_alpha(msh, anom=anomaly, background=1.0)
	alphaBaru = ms_baru['alpha']
	# print(ms_baru)

	startSolve(FP, exLine, alphaBaru) # Forward Problem
	inverseProblemSolver(FP, inputVol, msh, alphaBaru)	# Inverse Problem
print (FP)
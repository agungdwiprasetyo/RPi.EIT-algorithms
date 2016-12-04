#!/usr/bin/env python3

"""
Forward Problem, belum sampai ke Inverse Problem karena belum menggunakan salah satu algoritma rekonstruksi seperti JAC, BP, ataupun Greit
"""

import mesh
from mesh import quality # buat ngecek jumlah node dan element model citra
from eit.utils import eit_scan_lines
from eit.fem import forward
import numpy as np
import matplotlib.pyplot as plt

jumlahElektroda = 16

ms, elektroda = mesh.create(jumlahElektroda, h0=0.1) # init mesh
print (elektroda)

nodeXY = ms['node']
numElemen = ms['element']

quality.fstats(nodeXY, numElemen) # print jumlah node dan element

Mat1 = eit_scan_lines(16, 7) # membuat matriks diagonal yg nilainya 1 yg ukurannya sebesar 16*16
exLine = Mat1[0].ravel()
# print(exLine)

forwardProblem = forward(ms, elektroda) # memulai forward problem

while True:
	# x = float(input("Nilai x: "))
	# y = float(input("Nilai y: "))
	d = float(input("Nilai d: "))
	alp = float(input("Alpha: "))
	# ngeset anomali, nanti muncul di plot yang warna abu-abu, bertipe data Dictionary (json)
	anomaly = [{'x': 0, 'y': 0, 'd': d, 'alpha': alp}] # x dan y menyatakan posisi tengah dari anomali yg terbentuk, d menyatakan range
	ms_baru = mesh.set_alpha(ms, anom=anomaly, background=1.0)
	tri_perm = ms_baru['alpha']
	# print(ms_baru)

	ff,j = forwardProblem.solve_once(exLine,tri_perm) # ff masih terdapat bilangan imajiner
	ff = np.real(ff) # ff di-real kan
	# print(min(ff))

	vf = np.linspace(min(ff), max(ff), 32) # membagi array ff dari nilai minimal ke maksimal menjadi sebanyak 32 dengan selisih tiap2 nilai sama
	# print(vf)

	# plot
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.tricontour(nodeXY[:, 0], nodeXY[:, 1], numElemen, ff, vf, linewidth=0.5, cmap=plt.cm.viridis)
	ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], numElemen, np.real(tri_perm), edgecolors='k', shading='flat', alpha=0.5, cmap=plt.cm.Greys)
	ax1.plot(nodeXY[elektroda, 0], nodeXY[elektroda, 1], 'ro')
	ax1.set_title('Forward Problem')
	ax1.axis('equal')
	fig.set_size_inches(6, 4)
	plt.show()
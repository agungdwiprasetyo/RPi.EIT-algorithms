#!/usr/bin/env python3

"""
Forward Problem, belum sampai ke Inverse Problem karena belum menggunakan salah satu algoritma rekonstruksi seperti JAC, BP, ataupun Greit
"""

import mesh
from mesh.quality import fstats # buat ngecek jumlah node dan element model citra
from eit.utils import eit_scan_lines
from eit.fem import forward
import eit.jac as jac 
import numpy as np
import matplotlib.pyplot as plot

jumlahElektroda = 16

ms, elektroda = mesh.create(jumlahElektroda, h0=0.1) # init mesh
print (elektroda)

nodeXY = ms['node']
numElemen = ms['element']

fstats(nodeXY, numElemen) # print jumlah node dan element

Mat1 = eit_scan_lines(16, 7) # membuat matriks diagonal yg nilainya 1 yg ukurannya sebesar 16*16
exLine = Mat1[0].ravel()
# print(exLine)
forwardProblem = forward(ms, elektroda) # memulai forward problem


# Memulai proses pencitraan
x = 0.2
y = 0.5
d = 0.3
alp = 3
# ngeset anomali, nanti muncul di plot yang warna abu-abu, bertipe data Dictionary (json)
anomaly = [{'x': x, 'y': y, 'd': d, 'alpha': alp}] # x dan y menyatakan posisi tengah dari anomali yg terbentuk, d menyatakan range
ms_baru = mesh.set_alpha(ms, anom=anomaly, background=1.0)
tri_perm = ms_baru['alpha']
# print(ms_baru)

# mulai proses forward problem
ff,j = forwardProblem.solve_once(exLine,tri_perm) # ff masih terdapat bilangan imajiner
ff = np.real(ff) # ff di-real kan
# print(min(ff))
vf = np.linspace(min(ff), max(ff), 32) # membagi array ff dari nilai minimal ke maksimal menjadi sebanyak 32 dengan selisih tiap2 nilai sama
# print(vf)
# plot
fig = plot.figure()
ax1 = fig.add_subplot(111)
ax1.tricontour(nodeXY[:, 0], nodeXY[:, 1], numElemen, ff, vf, linewidth=0.5, cmap=plot.cm.viridis)
ax1.tripcolor(nodeXY[:, 0], nodeXY[:, 1], numElemen, np.real(tri_perm), edgecolors='k', shading='flat', alpha=0.5, cmap=plot.cm.Greys)
ax1.plot(nodeXY[elektroda, 0], nodeXY[elektroda, 1], 'ro')
ax1.set_title('Forward Problem')
ax1.axis('equal')
fig.set_size_inches(6, 4)
plot.show()

print("Sedang memproses inverse problem...")
# proses inverse problem
f0 = forwardProblem.solve(Mat1, step=1, perm=ms['alpha'])
f1 = forwardProblem.solve(Mat1, step=1, perm=tri_perm)
eit = jac.JAC(ms, elektroda, exMtx=Mat1, step=1,perm=1., parser='std',p=0.30, lamb=1e-4, method='kotre')
ds = eit.solve(f1.v, f0.v)
print("Sukses.")
# plot
fig = plot.figure()
plot.tripcolor(nodeXY[:, 0], nodeXY[:, 1], numElemen, np.real(ds),shading='flat', cmap=plot.cm.viridis)
plot.colorbar()
ax1.set_title('Inverse Problem')
plot.axis('equal')
fig.set_size_inches(6, 4)
plot.show()

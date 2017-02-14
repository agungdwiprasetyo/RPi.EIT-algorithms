# Python Electrical Impedance Tomography
Electrical Impedance Tomography (EIT) adalah  suatu konsep pencitraan dari distribusi resistivitas  listrik  internal  suatu objek  dengan  pengukuran  beda potensial  listrik  antar  elektrode yang  terhubung  dengan objek.  Teknik  ini  bekerja  dengan  cara  menginjeksikan  arus listrik  pada  objek  melalui elektrode yang terpasang pada permukaan objek.

Repositori ini berisi program untuk merekonstruksi citra pada EIT berdasaran data tegangan yang diperoleh dari alat EIT yang telah dibuat. Program ditulis dalam bahasa pemrograman Python. Versi python yang digunakan yaitu versi 3 keatas.

Permasalahan  dalam  rekonstruksi citra  pada  EIT  dapat  dipecah menjadi  dua  yaitu Forward  Problem dan Inverse Problem. Penyelesaian Forward Problem dapat dilakukan dengan Finite Element Method (FEM). Kemudian teknik rekonstruksi untuk bagian inverse problem dalam program ini yaitu menggunakan algoritma BP (Back Projection), JAC (Gauss-Newton solver), dan Greit (menggunakan metode distribusi).

Ada tambahan satu algoritma rekonstruksi untuk Inverse Problem, yaitu Simultaneous Algebraic Reconstruction Technique (SART) yang terdapat dalam folder ``` /SART ```. Algoritma ini akan diselesaikan dengan teknik *paralell processing* menggunakan GPU (Graphic Processing Unit). Tujuannya yaitu untuk membandingkan kecepatan serta layak atau tidak digunakan dalam pencitraan EIT.

## Requirements

Versi python yang digunakan yaitu Python 3.5, dan menggunakan sistem operasi Linux 64-bit. Adapun beberapa library yang dibutuhkan supaya dapat menjalankan program dalam repositori ini adalah sebagai berikut:
| Library  | Shell script |
| ---- | ---- |
| **numpy** | ```$ sudo apt-get install python3-numpy``` |
| **scipy** | ```$ sudo apt-get install python3-scipy``` |
| **matplotlib** | ```$ sudo apt-get install python3-matplotlib``` |
| **vispy** | ```$ sudo apt-get install python3-vispy``` |
| **pandas** | ```$ sudo apt-get install python3-pandas``` |
| **xarray** | ```$ sudo apt-get install python3-xarray``` |
| **distmesh** | ```$ sudo apt-get install python3-distmesh``` |


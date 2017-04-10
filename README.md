# Electrical Impedance Tomography
Electrical Impedance Tomography (EIT) adalah  suatu konsep pencitraan dari distribusi resistivitas  listrik  internal  suatu objek  dengan  pengukuran  beda potensial  listrik  antar  elektrode yang  terhubung  dengan objek.  Teknik  ini  bekerja  dengan  cara  menginjeksikan  arus listrik  pada  objek  melalui elektrode yang terpasang pada permukaan objek.

![EIT](https://github.com/agungdwiprasetyo/EIT/raw/master/pic/chickentiss.jpeg)

Repositori ini berisi program untuk merekonstruksi citra pada EIT berdasaran data tegangan yang diperoleh dari alat EIT yang telah dibuat. Program ditulis dalam bahasa pemrograman Python. Versi python yang digunakan yaitu versi 3 keatas.

Permasalahan  dalam  rekonstruksi citra  pada  EIT  dapat  dipecah menjadi  dua  yaitu Forward  Problem dan Inverse Problem. Penyelesaian Forward Problem dapat dilakukan dengan Finite Element Method (FEM). Kemudian teknik rekonstruksi untuk bagian inverse problem dalam program ini yaitu menggunakan algoritma Back Projection, JAC (Gauss-Newton/Jacobian solver), dan GREIT (menggunakan metode distribusi).

Ada tambahan satu algoritma rekonstruksi untuk Inverse Problem, yaitu Simultaneous Algebraic Reconstruction Technique (SART) yang terdapat dalam folder ``` /SART ```. Algoritma ini akan diselesaikan dengan teknik *paralell processing* menggunakan GPU (Graphic Processing Unit). Tujuannya yaitu untuk membandingkan kecepatan serta layak atau tidak digunakan dalam pencitraan EIT.

## Requirements

Versi python yang digunakan yaitu Python 3.5, dan menggunakan sistem operasi Linux 64-bit. Install ```python-pip``` dan ```python-dev```. 

```$ sudo apt-get install python3-pip python3-dev```

Kemudian beberapa library yang dibutuhkan supaya dapat menjalankan program dalam repositori ini adalah sebagai berikut:

| Library  | Command |
| ---- | ---- |
| **numpy** | ```$ sudo python3 -m pip install numpy``` |
| **scipy** | ```$ sudo python3 -m pip install scipy``` |
| **matplotlib** | ```$ sudo python3 -m pip install matplotlib``` |
| **vispy** | ```$ sudo python3 -m pip install vispy``` |
| **pandas** | ```$ sudo python3 -m pip install pandas``` |
| **xarray** | ```$ sudo python3 -m pip install xarray``` |


## Rancangan Sistem

Dalam perancangan sistem ini dibagi menjadi 3 bagian utama yaitu penyelesaian Forward Problem, penyelesaian Inverse Problem, dan front-end untuk visualisasi citra. Secara garis besar, rancangan sistem dapat digambarkan sebagai berikut:

![sistem](https://github.com/agungdwiprasetyo/EIT/raw/master/pic/desainsistem.jpg)

1. **Penyelesaian Forward Problem:**
Pada bagian ini, data akuisisi dari perangkat EIT akan diolah oleh sebuah single board computer. Single board computer yang digunakan yaitu Raspberry Pi. Hasil luaran dari bagian ini adalah data hasil pengolahan dengan penyelesaian Forward Problem yang menggunakan Finite Element Method (FEM). Data ini selanjutnya akan dikirim ke server untuk diolah lebih lanjut melalui koneksi internet.
2. **Penyelesaian Inverse Problem:** 
Penyelesaian Inverse Problem ini terjadi pada server. Data dari langkah pada penyelesaian Forward Problem sebelumnya akan digunakan untuk penyelesaian Inverse Problem. Data yang diolah dalam proses Inverse Problem ini sangat besar, sehingga dimungkinkan akan dilakukan pemrosesan paralel menggunakan GPU (*Graphics Processing Unit*) pada server. Hasil luaran dari bagian ini yaitu citra yang sudah terbentuk untuk selanjutnya akan ditampilkan pada aplikasi web.
3. **Front-end untuk visualisasi citra:** 
Bagian ini menggunakan aplikasi web untuk memvisualisasikan citra hasil rekonstruksi yang diperoleh dari proses Inverse Problem pada server. Front-end untuk membangun aplikasi web ini menggunakan AngularJS.

## Tahapan

Data yang digunakan berasal dari penelitian pengukuran tegangan pada tungkai hewan. Data dapat dilihat pada folder [/data](https://github.com/agungdwiprasetyo/EIT/tree/master/data). Objek dalam penelitian ini dapat disimulasikan dalam bentuk rangkaian resistor seperti gambar dibawah ini:

![resistor](https://github.com/agungdwiprasetyo/EIT/raw/master/pic/PhantomResistor.png)

Data tegangan diperoleh dengan mengukur tegangan pada semua elektroda yang terhubung dalam kulit objek, dalam hal ini tungkai hewan.

Apabila semua library yang dibutuhkan sudah terinstall, jalankan program **main.py** dengan mengetikkan perintah ```$ python3 main.py``` atau ```$ ./main.py```.  Maka proses yang sedang berjalan yaitu dapat dilihat pada gambar dibawah ini:

![proses](https://github.com/agungdwiprasetyo/EIT/raw/master/pic/Proses.png)

Pada proses diatas ditampilkan matriks Jacobian yang diperoleh dari proses Finite Element Method (FEM). Matriks Jacobian ini akan digunakan pada inverse problem untuk mengolah data yang diperoleh dari hasil pengukuran menjadi sebuah citra. Untuk saat ini, penyelesaian inverse problem menggunakan algoritma Back Projection.

Hasil akhir dari proses pencitraan pada EIT untuk data yang diperoleh dari hasil pengukuran tungkai hewan dapat ditunjukkan seperti gambar dibawah ini:

![hasil](https://github.com/agungdwiprasetyo/EIT/raw/master/pic/TesData-BP.png)
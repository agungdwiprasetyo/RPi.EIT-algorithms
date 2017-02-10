# Simultaneous Algebraic Reconstruction Technique

Tomographic reconstruction using the Simultaneous Algebraic Reconstruction Technique (SART) implemented in python.

Library yang dibutuhkan:
* ``` multiprocessing ```
* ``` matplotlib ```
* ``` numpy ```
* ``` scipy ```

Menghilangkan error ``` RuntimeError: dvipng was not able to process the following file``` ketika proses matplotlib pada python:

```sh
$ sudo apt-get install dvipng 
```

## Mulai

Jalankan program ``` main.py```, maka akan terjadi proses seperti berikut:
![proses](https://github.com/agungdwiprasetyo/EIT/raw/master/SART/process.png)

Gambar diatas merupakan proses dari algoritma SART. Dari setiap iterasi akan menghasilkan update citra terbaru. Citra hasil rekonstruksi yaitu seperti ditunjukkan pada gambar dibawah ini:
![proses](https://github.com/agungdwiprasetyo/EIT/raw/master/SART/sart_0.png)

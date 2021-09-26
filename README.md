# data-sains-fakhri
Untuk belajar Data Sains oleh Ahmad Dhiya Al Fakhri
# mencari SSE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#memasukkan data terlebih dahulu
x = [140,160,190,170,187,110,155,245,247,238,168,170,260,159,170]
y = [245,312,279,350,200,219,405,450,324,319,258,415,298,300,340]

#membuat grafik scatter
plt.xlabel("Luas Tanah")
plt.ylabel("Harga Rumah")
plt.xticks(np.arange(min(x),max(x)+20,20))
plt.yticks(np.arange(min(y),max(y)+20,20))
plt.scatter(x,y)
plt.grid(True)
plt.show()

#menjadikan list yang di csv menjadi array
#karena untuk menggunakan modul ini harus memakai data dari array
#h adalah variabel untuk menampung array dari x
h = [[x[i]]for i in range (len(x))]
#k adalah variabel untuk menampung array dari y
k = [[y[i]]for i in range (len(y))]

#memanggil fungsi Linear Regression
model=LinearRegression()
#membuat model fit (menyamakan titik x dan y) dari x dan y yang sudah dibentuk array
model.fit(h,k)
#membuat y predict
y_predict= model.predict(h)

#membuat grafik
#membuat label x dan y
plt.xlabel("Luas Tanah")
plt.ylabel("Harga Rumah")
#membuat angka angka yang muncul di grafik dengan gap/jarak 20
plt.xticks(np.arange(min(x),max(x)+20,20))
plt.yticks(np.arange(min(y),max(y)+20,20))
#membuat grafik menjadi grid
plt.grid(True)
#menggambar plot regresi linear
plt.plot(h,y_predict,color="b")
#membuat plotnya
plt.show()

#memanggil fungsi polynom derajat
satu= np.poly1d(np.polyfit(x,y,1))
dua= np.poly1d(np.polyfit(x,y,2))
tiga= np.poly1d(np.polyfit(x,y,3))
line= np.linspace(min(x),max(x))

plt.xlabel("Luas Tanah")
plt.ylabel("Harga Rumah")
plt.xticks(np.arange(min(x),max(x)+20,20))
plt.yticks(np.arange(min(y),max(y)+20,20))
plt.grid(True)
plt.plot(line,satu(line),color="y")
plt.plot(line,dua(line),color="g")
plt.plot(line,tiga(line),color="r")
plt.show()

sse=0
for j in range(0,len(y)):
    e2=(y[j]-y_predict[j])**2
    sse+=e2
print('Nilai SSE dari data diatas adalah = ', sse)

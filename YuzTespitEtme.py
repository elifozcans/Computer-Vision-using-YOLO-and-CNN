# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:01:52 2021

@author: Elif,İrem,İlayda,Sümeyye
"""

#Resimden Yüz Saptama
#FaceDetectionFromImageFiles
#opencv görüntü işlemede kullanılan çok güçlü bir kütüphanedir
#pip install opencv-python

import numpy as np
import cv2 #opencv kütüphanemizi import ettik
import matplotlib.pyplot as plt #NumPy için bir çizim kitaplığıdır
 
img = cv2.imread("C:/Users/EMRE/YOLO/model/faces.png")
img
img.shape #(w,h,3(RGB): Red,Green,Blue:3,resmin renkli olduğunu ifade eder; eğer renksiz bir resim olsaydı 1 olurdu)
plt.imshow(img) ; 
#Resmi renkliden gri hale getirdik;bunun amacı bilgisayarlar renksiz resimleri işlemede daha yüksek performans gösteriyor
gray_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
gray_scale
print(gray_scale.shape)
print(gray_scale.size)

#eğer resmi renkli olarak kullansaydık
#renksiz halde kullandığımız için büyük boyuttan tasarruf etmiş olduk
print(img.shape)
print(img.size)
plt.imshow(gray_scale) ;

#yüzleri tanımlayalım

face_cascade = cv2.CascadeClassifier("C:/Users/EMRE/YOLO/model/haarcascade_frontalface_default.xml")
#grayscale detect et, ve ölçek 1.1-1.4 arasında optimal çözümü veriyor,3. parametre olarak min komşu sayısı;
#(minumum kaç tane kare olsun ki ben yüzleri kesin olarak tanımlayayım:güvenilir olmasını istiyorsak değeri yüksek girebiliriz ama o zaman da yüzleri tanıyamayabiliyor)
#şimdilik 3 yaptık ama parametrelerle oynayarak anlayabiliriz
faces = face_cascade.detectMultiScale(gray_scale, 1.1,4) #1.1 ile dene 4 e çıkar(businesspeople)/1.1,3(people2)/1.1,4(faces)

faces.shape #(3,4) #faces a 4 değişken atamış
faces #Array deki her satır sayısı bize algınan kaç tane yüz olduğunu gösterir

#yüzleri tanımladık, peki yüz etrafındaki dikdörtgeni nasıl oluşturacağız?
#bunun için bir for döngüsü oluşturuyoruz
#döngüde faces kullanıyorum yukarıda bütün faceleri öğrendiğini varsayıyorum

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),4) #0,0,255 RGB renk kodu, 4 dikdörtgenin kalınlığı
    
cv2.imshow("face detection", img)
cv2.waitKey(0) 
cv2.destroyAllWindows() #bastığımızda kapatabilmek için 
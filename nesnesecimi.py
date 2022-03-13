# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:14:19 2021

@author: Elif,İrem,İlayda,Sümeyye
"""

#WARP PRESPECTIVE İLE ÇOKLU NESNE İÇEREN RESİMDEN İSTEDİĞİN
#NESNEYİ ÇEKİP ALMAK
import cv2
import numpy as np

img = cv2.imread("C:/Users/EMRE/YOLO/model/defterler2.png")

#Çekirdek matris görüntü işlemede kullanılan bir matristir.Bulanıklaştırma, kabartma, kenar algılama ve daha fazlası için kullanılır.
#Bizim amacımız kenar tespitinde, bu kenarları korumak ve diğer her şeyi atmak, çıkarmak istiyoruz. Bu nedenle, yüksek geçişli bir filtrenin eşdeğeri olan bir kernel oluşturmalıyız.
#Bir kernel bir nxn kare matristir ve n tek sayıdır. Çekirdek, dijital filtreye bağlıdır.3 x 3 ortalama filtre için kullanılan çekirdeği göstermektedir.
kernel = np.ones((3,3),np.uint8) #Numpy ile kernel matris tanımı #görselde yüz özelliklerini arar ve özellikleri karşılayan bölgeyi tespit edip işaretler.
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Resmi gri yaptık
cv2.imshow("Gray Image", imgGray)

width,height = 303,404
#pts1 = np.float32([[113,63],[164,60],[146,264],[201,258]],) #4.kitap
#pts1 = np.float32([[148,23],[247,53],[114,167],[210,193]],) #10 numaralı kart
pts1 = np.float32([[1125,63],[1371,71],[1133,439],[1371,445]],) #kedi desenli defter
#pts1 = np.float32([[613,531],[871,549],[613,931],[873,933]],) #çiçek desenli defter
#pts1 = np.float32([[133,555],[371,548],[132,924],[375,936]],) #yazılı defter
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) 
matrix = cv2.getPerspectiveTransform(pts1,pts2)
#warpPerspective ile Görüntü üzerinde bilgi toplamak istediğimiz noktaları sağlıyoruz.
imgOutput = cv2.warpPerspective(imgGray,matrix,(width,height)) 


cv2.imshow("Output",imgOutput) 

cv2.waitKey(0)
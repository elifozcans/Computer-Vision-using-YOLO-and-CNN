# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:28:40 2021

@author: Elif,İrem,İlayda,Sümeyye
"""
#pip install opencv-python
import cv2

face_cascade = cv2.CascadeClassifier(
    r"C:/Users/EMRE/YOLO/model/haarcascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(r"C:/Users/EMRE/YOLO/model/VIDEOS/India.mp4") 
while cap.isOpened():
    _,img = cap.read() #video tamamlanana kadar yüzleri tanımlayabilsin

    #img = cv2.imread(r"photos/1588088785877.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #videoyu renksiz hale getirdik

    faces = face_cascade.detectMultiScale(gray,1.2,5) #Videodaki yüzleri tespit edebilmesini sağladık 5:güvenilirlik düzeyi
# ölçek 1.1-1.4 arasında optimal çözümü veriyor

    for(x,y,w,h) in faces: #yüzlerin etrafına dikdörtgen kutular çizdik
        cv2.rectangle(img,(x,y),(x+w, y+h),(247,203,163),3) #3:dikdörtgenin kalınlığı #163:KIRMIZI 203:YEŞİL 247:MAVİ
    cv2.imshow('img',img) #ortaya çıkan kareyi göstermek için kullandık
    
    if cv2.waitKey(1) & 0xFF == ord('q'):    #Çıkış yapmak için Q'ya bas
        break             

cap.release() #videoyu gösterebilmek için 
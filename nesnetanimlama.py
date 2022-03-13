# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:50:40 2021

@author: Elif,İrem,İlayda,Sümeyye
"""
#YOLO: "YOU ONLY LOOK ONCE"
import cv2 
import numpy as np

img = cv2.imread("C:/Users/EMRE/YOLO/model/childroom.jpg")
print(img)

#resmin enini ve boyunu belirtiyoruz
img_width = img.shape[1] #1=1130
img_height = img.shape[0] #0=853

#Üzerinde işlem yapılabilmesi için görüntüyü 4 boyutlu hale getiriyoruz
#blob işlevi, ortalama çıkarma, normalleştirme ve kanal değiştirmeden sonra girdi resmimiz olan bir blob döndürür.
img_blob = cv2.dnn.blobFromImage(img, 1/255,(416,416), swapRB = True)
#YOLO da 1/255 standart olarak kullanılıyor(mean değer)
#416,416 ise YOLO için indirdiğimiz dosyaların değerleri
#OpenCV, görüntülerin BGR kanal sıralamasında olduğunu varsayar; ancak "ortalama" değer, RGB sırasını kullandığımızı varsayar. 
#Bu tutarsızlığı gidermek için, bu değeri "True" olarak ayarlayarak görüntüdeki R ve B kanallarını değiştirebiliriz.
#Varsayılan olarak OpenCV bu kanal değişimini bizim için gerçekleştirir.

#Modelin hazır olarak bize sunduğu isimlerini ekliyoruz
#load YOLO #dnn:deep neural network
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
          "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
          "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
          "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]
#Çizeceğimiz boxların renklerini belirliyoruz
colors =["255,255,0","0,255,0","255,0,255","0,255,255","0,0,255"]
#Virgülden sonra her bir  255 i oku o yüzden virgül koyduk "," den sonra her birini parçala
colors =[np.array(color.split(",")).astype("int") for color in colors]
#Colors değişkenini bir dizi içerisine koyduk
colors = np.array(colors)
#biz sadece 5 adet color girdik fakat bizde 80 adet etiket bu da 80 adet nesne demek bu yüzden renkleri çoğaltıyoruz
#20 satır aşağı 1 satır yana çoğalt: amacımız colors değişkenlerini çoğaltmak
colors =np.tile(colors,(20,1))

#indirdiğimiz dosyaları modele dahil ediyoruz; cfg ve weights dosyalarımızın yolunu dahil ediyoruz
model = cv2.dnn.readNetFromDarknet ("C:/Users/EMRE/YOLO/model/yolov3.cfg","C:/Users/EMRE/YOLO/model/yolov3.weights")
layers = model.getLayerNames() #içerisinde birçok katman var fakat biz YOLO katmanlarını dahil ediyoruz
#hangi katmanları dahil edeceğimizi görmek için
#model.getUnconnectedOutLayers() --> konsol kısmına yazılacak ve "enter" #200,227,254
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()] #bize YOLO lazım old.için -1 azaltıyoruz
#0 indeks değerinin amacı ise katmanlar içerisinde bir döngü oluşturabilmek:her seferinde içeride bir döngü oluşturup gerekli olan katmanı seçmesini sağlıyoruz
#modelimize 4 boyutlu resmi gönderelim
model.setInput(img_blob)
#çıktı katmanları için
detection_layers = model.forward(output_layer) #value:matrisler oluştu bu matrisler ve içerisindeki değerlerle de tek tek dolaşmamız gerekiyor bunun için bir for döngüsü oluşturuyoruz


# önce 86.satır
ids_list = []
boxes_list = []
confidences_list = []

#her bir matris for döngüsü oluşturuyoruz

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection [5:]#bulunan nesnenin güven skorunu oluşturuyoruz #5 den sonrası için #bu skorlar arasında en büyük değer en tahmin edilen nesne olacak
        predicted_id = np.argmax(scores) #skorlar arasında en yüksek güven aralığını tutmak gerekiyor bunun için id oluşturuyoruz
        confidence = scores[predicted_id] #güven skoru değişkeni oluşturuyoruz ve skorları atıyoruz
        
        #ve bir güven aralağı oluşturuyoruz
        if confidence > 0.50  : #güven aralığında %50 dan fazla bulduğu nesneyi çizmesini istiyoruz
         label = labels [predicted_id] #label ları bulduğu id lere atıyoruz
         boundary_box = object_detection[0:4] * np.array ([img_width,img_height,img_width,img_height]) #kutuları oluşturuyoruz #burada oluşan değerler bizim için yeterli olmadığı için bu değerleri büyütüp anlamlı hale getiriyoruz
                                              #bunun için numpy kütüp. kullanarak ilk başta oluş.resmin eni ve boyu ile 2 kez çarpıyoruz
         (box_center_x , box_center_y , box_width , box_height) = boundary_box.astype("int")  #tespit edeceğimiz kutuları oluşturmak için bazı değerler girmemiz gerekiyor ve bu değerler int olmalı(önceden float şeklindeyi)
         
         #kutu merkezlerini oluşturuyoruz:başlangıç noktaları
         start_x = int(box_center_x - (box_width/2))
         start_y = int(box_center_y - (box_height/2))
         
         #önce 3 değişken oluşturduk (ids,confidences list ve boxes list): amacımız sadece emin olduğu nesneleri kare içine alsın
         ids_list.append(predicted_id) 
         confidences_list.append(float(confidence))
         boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
         
        
         #bu komut bütün yüksek güvenirliğe sahip dikdörtgenleri bir liste biçiminde geri döndürüyor 0.5:güven skoru 0.4:eşik değeri
         #en son for döngüsünden çıkarak bulduğumuz max güvenirliğe sahip box ları max id sinin içerisine sakladık
max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)
         
         #for döngüsü içerisinde önceden tespit ettiğimiz değişkenleri bu dizilerin içerisine yolladık
for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list [max_class_id]
             
    start_x = box [0]
    start_y = box [1]
    box_width = box [2]
    box_height = box [2]
            
            
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
             
             
         
         #kutu merkezlerini oluşturuyoruz:bitiş noktaları
    end_x = start_x + box_width
    end_y = start_y + box_height
         
         #kutular için renk ataması yapıyoruz
    box_color = colors[predicted_id]
    box_color = [int (each) for each in box_color]
         
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2) #2:kutu kalınlığı
         #kutu üzerine yazılacak label leri belirliyoruz (-20 sağına yazar, 0.5 yazının boyutu ,1 yazı kalınlığı)
    cv2.putText(img,label,(start_x,start_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,box_color,1)
    #ilk değişkenimiz resim,2.değişken üzerine tanımlanacak isim,3.değişken yazıyı nerede yazdıracağımız,4. değişken yazı tipi
        
         
cv2.imshow("Tespit Ekranı", img )













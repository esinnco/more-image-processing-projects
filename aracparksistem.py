import cv2
import numpy as np
import pickle

rectW,rectH=107,48
#video dosyası açma
cap=cv2.VideoCapture('carPark.mp4')

with open('carParkPos','rb') as f:
    #load fonksiyonu dosyanın veri tipini değiştirmemizi sağlar.
    posList=pickle.load(f)
frame_counter = 0
def check(imgPro):
    spaceCount=0
    for pos in posList:
        x,y=pos
        crop=imgPro[y:y+rectH,x:x+rectW]
        count=cv2.countNonZero(crop)
        #araba yoksa yeşil, araba varsa kırmızı olacak.
        if count<900:
            spaceCount+=1
            color=(0,255,0)
            thick=5
        else:
            color=(0,0,255)
            thick=2

        cv2.rectangle(img,pos,(x+rectW,y+rectH),color,thick)
    #rectangle:herhangi bir görüntüye dikdörtgen çizdirme
    cv2.rectangle(img,(45,30),(250,75),(180,0,180),-1)
    #kaç park yerinin boş olduğunu gösteren sayaç yazdırma
    cv2.putText(img,f'Free: {spaceCount}/{len(posList)}',(50,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

while True:
    _,img=cap.read()
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    #resmi gri tonlamalı yapma    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gaussian blur: düzleştirme filtresi, resim yumuşatma
    blur=cv2.GaussianBlur(gray,(3,3),1)
    Thre=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    #median blur:ortalama yumuşatma, blurlama filtresi
    blur=cv2.medianBlur(Thre,5)
    kernel=np.ones((3,3),np.uint8)
    #dilate() operatörü ile verilen alan içindeki sınırları genişlettik
    dilate=cv2.dilate(blur,kernel,iterations=1)
    check(dilate)
    
    #görüntü yazdırma
    cv2.imshow("Image",img)
    cv2.waitKey(10)

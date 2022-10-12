import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

#resim açma
img=cv2.imread("cat.jpg")
#bilgisayarın rahat okuması için grileştirme
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#bu fonksiyon yüz belirlemek için kullanılır.
faces=face_cascade.detectMultiScale(gray, scaleFactor=1.01,minNeighbors=5)


for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    #rectangle ile yüzü belirteceğimiz dikdörtgeni çizdik.

cv2.imshow("cat face",img)
cv2.waitKey()
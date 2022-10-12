
import cv2
import mediapipe as mp
import math


class HandDetector:
    #Mediapipe kütüphanesini kullanarak elleri tespit eder.İki parmak arasındaki mesafe veya kaç parmağın yukarıda olduğunu bulur.

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        #maxHands parametresi :algılanacak max el sayısı
        #detectionCon : min algılama eşiği
        #minTrackCon : min izleme eşiği
        #self , bir sınıfın nesnelerini tespit etmek için kullanılır.
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        #Bir BGR görüntüsünde el bulma.
        
        #BGRyi RGBye çevirme
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(imgRGB)
        
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands
    
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList=[]
        bbox=[]
        
        self.lmList = []
        print(self.lmList)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
       
                self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        xmin, xmax = min(xList,default=0), max(xList,default=0)
        ymin, ymax = min(yList,default=0), max(yList,default=0)
        bbox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),(bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        
        return self.lmList, bbox


    def fingersUp(self, myHand):
        #Kaç parmağın açık olduğunu bulur ve bir listede döner.
        #Sol ve sağ eli ayrı değerlendirir
        #return: Hangi parmakların yukarıda olduğu listesi
        
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            #baş parmak
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # diğer 4 parmak
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        #Konumlarına göre iki yer işareti arasındaki mesafeyi bul.
        
        # p1: Nokta1
        # p2: Nokta2
        #img: Çizilecek resim.
        #draw: Çıktıyı görüntü üzerine çizmek için bayrak.
        #return: Noktalar arasındaki mesafe
                 

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def main():
    #webcam'i açma
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        
        success, img = cap.read()
        #eli ve işaretlerini bul(draw ile birlikte)
        hands, img = detector.findHands(img)  
        

        if hands:
            # el 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"] 
            bbox1 = hand1["bbox"]  
            centerPoint1 = hand1['center'] 
            handType1 = hand1["type"]  # sağ-sol el

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                # el 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # mediapipe el projesindeki 21 noktalı el bölgesinin listesi
                bbox2 = hand2["bbox"]  
                centerPoint2 = hand2['center'] 
                handType2 = hand2["type"]  

                fingers2 = detector.fingersUp(hand2)

                #iki el arasındaki uzaklığı bul.Aynı el üzerinde de olabilir.
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  
                
        # görüntüyü yazdır
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(0)
cap.set(3, 550)
cap.set(4, 550)

totalMoney = 0


myColorFinder = ColorFinder(True)
# Custom Orange Color
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}



def empty(a):
    pass

cv2.namedWindow("settings")
cv2.resizeWindow("settings",640,240)
cv2.createTrackbar("Threshold1","settings",219,255,empty)
cv2.createTrackbar("Threshold2","settings",233,255,empty)



def preProcessing(img):

    imgPre = cv2.GaussianBlur(img,(5,5),3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "settings")
    imgPre = cv2.Canny(imgPre,50,150)
    imgPre = cv2.Canny(imgPre,thresh1,thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

while True:
    success, img = cap.read()
    imgPre = preProcessing(img)
    imgcontours, conFound = cvzone.findContours(img,imgPre,minArea=20)





    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if len(approx)>5:
                area = contour['area']

                imgColor, _ = myColorFinder.update(img, hsvVals)


                if area<2050:
                    totalMoney +=5
                elif 2050<area<2500:
                    totalMoney +=1
                else:
                    totalMoney +=2
    print(totalMoney)
    imgStacked = cvzone.stackImages([img, imgPre, imgcontours], 2, 1)
    cvzone.putTextRect(imgStacked,f'Rs.{totalMoney}',(50, 50))

    cv2.imshow("Image", imgStacked)
    cv2.imshow("ImgColor", imgColor)

    cv2.waitKey(1)







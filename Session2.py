
import numpy as np
import cv2


def task1():
    image=cv2.imread('austria.jpg')
    grayImage = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayImage[i][j] = int(image[i][j][0]*0.2126 + image[i][j][1]*0.7152 + image[i][j][2] * 0.0722)
    grayImage.astype(np.uint8)
    cv2.imshow('Image1',image)
    cv2.imshow('Image2',grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows
################################################################################################################
# task1()  



def nothing(x):
    pass
def task2():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Trackbars')

    cv2.createTrackbar('Low_H', 'Trackbars',0,100,nothing)
    cv2.createTrackbar('Low_S', 'Trackbars',0,150,nothing)
    cv2.createTrackbar('Low_V', 'Trackbars',0,150,nothing)
    cv2.createTrackbar('Upper_H', 'Trackbars',100,179,nothing)
    cv2.createTrackbar('Upper_S', 'Trackbars',150,255,nothing)
    cv2.createTrackbar('Upper_V', 'Trackbars',150,255,nothing)

    while True:
        ret, frame = cap.read()        
        frame=cv2.resize(frame,(400,400))
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        LowH = cv2.getTrackbarPos('Low_H','Trackbars')
        LowS = cv2.getTrackbarPos('Low_S','Trackbars')
        LowV = cv2.getTrackbarPos('Low_V','Trackbars')
        HighH = cv2.getTrackbarPos('Upper_H','Trackbars')
        HighS = cv2.getTrackbarPos('Upper_S','Trackbars')
        HIghV = cv2.getTrackbarPos('Upper_V','Trackbars')

        lower = np.array([LowH,LowS,LowV])
        upper = np.array([HighH,HighS,HIghV])

        mask = cv2.inRange(hsv,lower, upper)
        result = cv2.bitwise_and(frame,frame,mask = mask)

        cv2.imshow('Frame',frame)
        cv2.imshow('Mask',mask)
        cv2.imshow('Result',result)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
################################################################################################################
# task2()  












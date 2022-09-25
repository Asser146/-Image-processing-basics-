
import numpy as np
import cv2


def detect(image):
    low_dark_red = np.array([130, 15, 0])
    high_dark_red = np.array([255, 255, 255])
    low_dark_yellow = np.array([22, 93, 0])
    high_dark_yellow = np.array([45, 255, 255])
    hsv_img= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1= cv2.inRange(hsv_img,low_dark_yellow,high_dark_yellow)
    mask2= cv2.inRange(hsv_img,low_dark_red,high_dark_red)
    YellowDetect= cv2.bitwise_and(image,image,mask=mask1)
    RedDetect= cv2.bitwise_and(image,image,mask=mask2)
    if np.sum(YellowDetect)>600:
        return True,YellowDetect
    if np.sum(RedDetect)>600:
        return True,RedDetect
    else:
        return False,image



def split(image):
    left_part=cv2.resize(image[:, 0:image.shape[1]//3],(360,360))
    right_part=cv2.resize(image[:,(image.shape[1]*2)//3:],(360,360))
    center_part=cv2.resize(image[:,image.shape[1]//3:(image.shape[1]*2)//3],(360,360))
    dc,Centerimage=detect(center_part)
    dl,Limage=detect(left_part)
    dr,Rimage=detect(right_part)
    if dc:
        cv2.imshow('Shape detected in Center part',Centerimage)
    if dr:
        cv2.imshow('Shape detected in Right part',Rimage)
    if dl:
        cv2.imshow('Shape detected in Left part',Limage)















image=cv2.imread('test1.png')
image=cv2.resize(image,(360,360))
point1= None
point2= None
CroppedImage= None
detected=False
def crop(event,x,y,flags,param): 
    global point1,point2,CroppedImage
    if(event == cv2.EVENT_LBUTTONDBLCLK):  
            print(f'New point is {(x,y)}')
            if point1 is None:
                point1= (x,y)
                print('Point 1  done')
            elif point2 is None:
                point2= (x,y)
                minX= min(point1[0],point2[0])
                maxX= max(point1[0],point2[0])
                minY= min(point1[1],point2[1])
                maxY= max(point1[1],point2[1])
                if minX != maxX and minY != maxY:
                    CroppedImage= image[minY:maxY, minX:maxX]   
                point1= None
                point2= None
cv2.namedWindow('Full image')  
cv2.setMouseCallback('Full image',crop)  
try:
    while(1):  
        cv2.imshow('Full image',image)  
        if CroppedImage is not None:
            cv2.imshow('Cropped Image',CroppedImage)  
            while(not detected):
                split(CroppedImage)
                detected=True
        if cv2.waitKey(1) & 0xFF == 27:  
            break 
except Exception as e:
    print(e)
cv2.destroyAllWindows()








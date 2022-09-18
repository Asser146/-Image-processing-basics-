
import cv2
import numpy as np

def task1():
    austria=cv2.imread('austria.jpg',0)
    maldives=cv2.imread('maldives.jpg',0)
    santorini=cv2.imread('santorini.jpg',0)
    switzerland=cv2.imread('switzerland.jpg',0)

    maldives=cv2.resize(maldives,(austria.shape[1],austria.shape[0]))
    santorini=cv2.resize(santorini,(austria.shape[1],austria.shape[0]))
    switzerland=cv2.resize(switzerland,(austria.shape[1],austria.shape[0]))
  
    image= np.zeros((austria.shape[0]*2,austria.shape[1]*2), dtype=np.uint8)
    image[:image.shape[0]//2,:image.shape[1]//2]+=austria
    image[:image.shape[0]//2,image.shape[1]//2:]+=switzerland
    image[image.shape[0]//2:,:image.shape[1]//2]+=santorini
    image[image.shape[0]//2:,image.shape[1]//2:]+=maldives


    cv2.imshow('Group image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# task1()



def task2():
    img1= cv2.resize(cv2.imread('Word.jpg',0), (360,360))        
    img2= cv2.resize(cv2.imread('E-letter.jpg',0), (360,360))
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Result',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# task2()



def task3():
    image = cv2.imread('austria.jpg',0)
    thresh=100
    image = image.astype(int)
    image[image<=thresh]= 0
    image=image.astype(np.uint8)
    cv2.imshow('Result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

 # task3()


def task4():
    kernel_sharp= np.array([
    [ 0, -1, 0],
    [-1, 5, -1],
    [ 0, -1, 0]
    ])
    image = cv2.imread('cat2.jpeg',0)
    SharpImage= cv2.filter2D(image,-1, kernel_sharp)
    cv2.imshow('Original',image)
    cv2.imshow('Sharp',SharpImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 # task4()
 
def task5():
    towers= cv2.resize(cv2.imread('towers.jpg',0), (360,360))
    BlurredTowers= cv2.GaussianBlur(towers, (5,5), 0)
    CannyEdges = cv2.Canny(BlurredTowers, threshold1=80, threshold2=220)
    cv2.imshow('Original',towers)
    cv2.imshow('Blurred',BlurredTowers)
    cv2.imshow('Canny Edges',CannyEdges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# task5()





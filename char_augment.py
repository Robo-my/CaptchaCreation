import numpy as np
import cv2
import random
import os


def addWhite(img,pixels):
    for i in range(random.randint(8, 20)):
        rand_int = random.randint(0, len(pixels)-1)
        img = cv2.circle(img,(pixels[rand_int][1],pixels[rand_int][0]), 4, (255,255,255), -1)
        return(img) 
def addBlack(img,pixels):
    for i in range(random.randint(8, 20)):
        rand_int = random.randint(0, len(pixels)-1)
        img = cv2.circle(img,(pixels[rand_int][1],pixels[rand_int][0]), 4, (0,0,0), -1)
        return(img)         
        
def morph_transform(img):
    i= random.randint(0, 1)
    if(i):
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = random.randint(0, 3))
        return(erosion)
    else:
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = random.randint(0, 2))
        return(dilation)      
        



charDataset = os.listdir('charDataset')


for char in charDataset:
    os.mkdir('charDataset_augment/'+ str(char)[:-4])
    number =25
    for i in range(number):
        img = cv2.imread('charDataset/'+str(char))
        edges = cv2.Canny(img, 100, 100)
        pixels = np.argwhere(edges == 255)
        white = addWhite(img,pixels)
        black = addBlack(white,pixels)
        morp = morph_transform(black)
        cv2.imwrite('charDataset_augment/'+ str(char)[:-4]+'/'+str(i)+'.jpg',morp)
    
    

#%%


  
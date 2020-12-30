# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:33:42 2020

Nail segmentation - hand images

Images are taken from a folder called images that is place in the same 
directory that this file

Results are store in a folder called results that is place in the same 
directory that this file

@author: mateo
"""

import cv2
from  matplotlib import pyplot as plt
import numpy as np
import os
import math

def FindColors(img, min_U, max_U, min_V, max_V, kernel_size):
    """
    
    Given a YUV image and a maximun and minumun value for the two 
    chrominance components U and V extract the mask that represent the 
    pixels in the image satisfiying the two chrominance components boundaries.

    Parameters
    ----------
    img : YUV image 
    min_U : INT
        Minimum value of U component allowed
    max_U : INT
        Maximun value of U component allowed
    min_V : INT
        Minimum value of V component allowed
    max_V : INT
        Maximun value of V component allowed
    kernel_size : INT
         Kernel size defining the morphological closing inside the function

    Returns
    -------
    Black and white mask representing the regions satisfying the two 
    chrominance components boundaries.
    
    """
    img_Area = img.shape[0]*img.shape[1]
    
    (NI_thresh_U, NI_blackAndWhiteImage_U) = cv2.threshold(img[:,:,1], min_U, 255, cv2.THRESH_BINARY)
    (I_thresh_U, I_blackAndWhiteImage_U) = cv2.threshold(img[:,:,1], max_U, 255, cv2.THRESH_BINARY_INV)
    blackAndWhiteImage_U = cv2.bitwise_and(I_blackAndWhiteImage_U, NI_blackAndWhiteImage_U)
        
    (NI_thresh_V, NI_blackAndWhiteImage_V) = cv2.threshold(img[:,:,2], min_V, 255, cv2.THRESH_BINARY)
    (I_thresh_V, I_blackAndWhiteImage_V) = cv2.threshold(img[:,:,2], max_V, 255, cv2.THRESH_BINARY_INV)
    blackAndWhiteImage_V = cv2.bitwise_and(I_blackAndWhiteImage_V, NI_blackAndWhiteImage_V)
    
    # cv2.imshow('U', blackAndWhiteImage_U)
    # cv2.imshow('V', blackAndWhiteImage_V)
    
    blackAndWhiteImage_UV = cv2.bitwise_and(blackAndWhiteImage_V, blackAndWhiteImage_U)
    # cv2.imshow('UV', blackAndWhiteImage_UV)
    
    blackAndWhiteImage_Areas = blackAndWhiteImage_UV.sum()/255
    if (blackAndWhiteImage_Areas == 0):
        ratio = 22
        # print("Inf")
    else:
        ratio = int(round(img_Area/blackAndWhiteImage_Areas))
        # print(ratio)
    if(ratio <= 21):
        #Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        closing = cv2.morphologyEx(blackAndWhiteImage_UV, cv2.MORPH_CLOSE, kernel, iterations = 1)
        # cv2.imshow('Closing', closing)
        return(closing)
    else:
        return(np.full((img.shape[0],img.shape[1]),255).astype(np.uint8))
 
def ConnectedComponents(blackAndWhiteMask, status = "above mean", min_size = 0, max_size = 0):
    """
    
    For a given mask the connected components that meet certain criteria 
    are chosen
    
    Parameters
    ----------
    blackAndWhiteImage : Mask representing in withe the components of the 
    image
    status : STRING, optional
       Defines what connected components are going to stay and witch not

    Returns
    -------
    New mask specifying the connected components meeting the given criteria
    
    """
    def Above(img, min_size):
        #For every component in the image, you keep it only if it's above min_size
        for i in range(0, n_components):
            if Non_Background_Stats[i] < min_size:
                img[labels == i +  1] = 0
        return(img)
    def Below(img, max_size):
        #For every component in the image, you keep it only if it's below max_size
        for i in range(0, n_components):
            if Non_Background_Stats[i] > max_size:
                img[labels == i +  1] = 0
        return(img)
    # your answer image
    img = blackAndWhiteMask
    #find all your connected components 
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhiteMask, connectivity=8)
    #Taking out the background
    Non_Background_Stats = stats[1:, -1]
    n_components = n_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    if(status == "above mean"):
        min_size = Non_Background_Stats.mean()
        img = Above(img, min_size)
    elif(status == "above max"):
        min_size = Non_Background_Stats.max()  
        img = Above(img, min_size)
    elif(status == "below mean"):
        max_size =  Non_Background_Stats.mean()
        img = Below(img, max_size)
    else:
         min_size = Non_Background_Stats.mean()/2
         img_min = Above(img, min_size)
         img_max = Below(img, max_size)
         img = cv2.bitwise_or(img_min, img_max)  
        
        
    return(img)

directory = os.fsencode('images')
for count,file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    img_in = cv2.imread('images/'+filename, 1)
    label = cv2.imread('labels/'+filename, 0)
    # cv2.imshow('Image', img)
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
    # create a CLAHE object (Arguments are optional) -> Adaptive Histogram Equalization 
    
    min_U = 90
    max_U = 125
    min_V = 145
    max_V = 170
        
    #Find color clusters 
    color_clusters =  FindColors(img, min_U, max_U, min_V, max_V, 100)
    
    #find all your connected components above averange
    img2 = ConnectedComponents(color_clusters.copy(), "above max")
    img2_Areas = img2.sum()/255
    # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    im_out = cv2.bitwise_and(img, img, mask = img2)  
    #cv2.imwrite('Results/Image'+str(count)+'.jpg',im_out)  
    # cv2.imshow('img Out', im_out)   
     
    # Otsu's thresholding 
    ret,th = cv2.threshold(im_out[:,:,0].astype(np.uint8),10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_Area = th.sum()/255
    if (img2_Areas/2 > th_Area):
        pass
    else:
        th = cv2.bitwise_xor(img2, th)  
        th_Area = th.sum()/255
    
    img3 = ConnectedComponents(th.copy(), "other", 0, img2_Areas/28)
    
    cv2.imwrite('Results/Image'+str(count)+'.jpg', img3) 
    cv2.imwrite('Results/Image-'+str(count)+'.jpg', im_out) 
     
    label_area =  label.sum()/255
    img3_area = img3.sum()/255
    intersection = cv2.bitwise_and(label, img3)  
    union = cv2.bitwise_or(label, img3)  
    intersection_area = intersection.sum()/255
    union_area = union.sum()/255
    IoU = intersection_area/union_area
    Dice = (2*intersection_area)/(label_area + img3_area)
    print(filename,"IoU: ",round(IoU,3)," Dice: ",round(Dice,3))
    
    
    
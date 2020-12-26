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
import time

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
        print("Inf")
    else:
        ratio = int(round(img_Area/blackAndWhiteImage_Areas))
        print(ratio)
    if(ratio <= 21):
        #Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        closing = cv2.morphologyEx(blackAndWhiteImage_UV, cv2.MORPH_CLOSE, kernel, iterations = 1)
        # cv2.imshow('Closing', closing)
        return(closing)
    else:
        return(np.full((img.shape[0],img.shape[1]),255).astype(np.uint8))
 
def ConnectedComponents(blackAndWhiteMask, status = "above mean"):
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
        pass 
    return(img)

directory = os.fsencode('images')
for count,file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    
    # filename = 'd6303a64-db67-11e8-9658-0242ac1c0002.jpg'
    img_in = cv2.imread('images/'+filename, 1)
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image', img)
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
    # create a CLAHE object (Arguments are optional) -> Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    img[:,:,0] = clahe.apply(img[:,:,0]) # Luminance Y channel
    # cv2.imshow('Equalized Image', img)
    
    img_Area = img.shape[0]*img.shape[1]
    img_x_size = img.shape[1]
    img_y_size = img.shape[0]
    #print("img_Area: ", img_Area)
    
    #Convolutions (Image Filtering) and kernel  -> Averaging
    #blur = cv2.blur(img,(10,10))
    blur = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)  
    #blur[:,:,2] = clahe.apply(blur[:,:,2]) 
    blur = cv2.bilateralFilter(blur,5,51,51) #Bilateral Filtering
      
    #Edge detection
    min_threshold = 0.1 * blur.mean()
    max_threshold = 0.6 * blur.mean()
    edges = cv2.Canny(blur, 0, max_threshold, apertureSize = 3)
    #cv2.imshow('Edges', edges)
    #cv2.imwrite('Results/Edges'+str(count)+'.jpg',edges)  
    
    #Histogram
    # plt.figure(0)
    # color = ('b','g','r') # b==y g==u and r==v
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([img],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.xticks(np.arange(256, step=10))
    # plt.show()
    # hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    # plt.hist(gray.ravel(),256,[0,256])
    
    min_U = 90
    max_U = 125
    min_V = 145
    max_V = 170
        
    #Find color clusters 
    color_clusters =  FindColors(img, min_U, max_U, min_V, max_V, 100)
    
    #find all your connected components above averange
    img2 = ConnectedComponents(color_clusters, "above max")
    
    # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    im_out = cv2.bitwise_and(img, img, mask = img2)
    #cv2.imwrite('Results/ImageF'+str(count)+'.jpg',im_out)  
    # cv2.imshow('img Out', im_out)
    
    
    #Histogram
    # plt.figure(1)
    color = ('b','g','r')
    max_values = []
    for i,col in enumerate(color):
        histr = cv2.calcHist([im_out],[i],None,[256],[10,256])
        max_values.append(histr.argmax())
    #     plt.plot(histr,color = col)
    #     plt.xlim([1,256])
    # plt.xticks(np.arange(256, step=10))
    # plt.show()
    
    
    # #Morphological Transformations for edge
    # edges = cv2.bitwise_and(edges, img2)
    # edge_thickness = int(blackAndWhiteImage_Areas*0.002/100)+1
    # print(edge_thickness)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(edge_thickness,edge_thickness))
    # edge_dilated = cv2.dilate(edges, kernel)   
    # # cv2.imshow('Edge dilatated', edge_dilated)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    # edge_delimiter = cv2.dilate(edges, kernel)
    
    # #(I XOR E) and I
    # I_xor_E = cv2.bitwise_xor(img2, edge_dilated)
    # regions = cv2.bitwise_and(I_xor_E, img2)
    
    # im_out = cv2.bitwise_and(img_in, img_in, mask = regions)
     
    
    # calculate moments of binary image
    M = cv2.moments(img2) 
    # calculate x,y coordinate of center   
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    bestX = cX
    bestY = cY
    for x in range(100, img_x_size - 100):
        actual_U = abs(im_out[cY,x,1] - max_values[1])
        best_U = abs(im_out[bestY, bestX, 1] -max_values[1])
        actual_V = abs(im_out[cY,x,2] - max_values[2])
        best_V = abs(im_out[bestY, bestX, 2] -max_values[2])
        if (actual_U < best_U and actual_V < best_V):
            bestX = x
    for y in range(100, img_y_size - 100):
        actual_U = abs(im_out[y,cX,1] - max_values[1])
        best_U = abs(im_out[bestY, bestX, 1] -max_values[1])
        actual_V = abs(im_out[y,cX,2] - max_values[2])
        best_V = abs(im_out[bestY, bestX, 2] -max_values[2])
        if (actual_U < best_U and actual_V < best_V):
            bestY = y
            bestX = cX
    
    #im_out = cv2.bitwise_and(img_in, img_in, mask = img2)
    # cv2.circle(im_out, (bestX, bestY), 3, (255, 255, 255), -1)
    # cv2.imwrite('Results/Region'+str(count)+'.jpg',im_out)
    
    skin_U = im_out[bestY, bestX, 1]
    skin_V = im_out[bestY, bestX, 2]
    
    min_U = skin_U - 10
    max_U = skin_U + 10
    min_V = skin_V - 10
    max_V = skin_V + 10

    #Find color clusters 
    color_clusters =  FindColors(img, min_U, max_U, min_V, max_V, 1)   
    
    
    img3 = cv2.bitwise_xor(img2, color_clusters)
    im_out = cv2.bitwise_and(img_in, img_in, mask = img3)
    cv2.imwrite('results/Image-'+str(count)+'.jpg',im_out)
    
    

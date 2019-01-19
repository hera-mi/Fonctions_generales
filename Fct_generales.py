# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:12:27 2019

@author: villa
"""
from scipy import ndimage
import numpy as np
from skimage.transform import rotate
import math


#Fonctions génerales

def gradlap():
    kx=np.zeros((3,3))
    kx[1][0],kx[1][2]=1,-1
    ky=np.transpose(kx)
    klap=np.ones((3,3))
    klap[1][1]=-8
    return(kx,ky,klap)
    
def gaussianKernel(hs,sig):
    kernel=np.zeros((2*hs+1 ,2*hs+1))
    for n in range(2*hs+1):
        for p in range(2*hs+1):
            kernel[n][p]=math.exp(-((n-hs)**2+(p-hs)**2)/2/sig**2)
    
    return kernel / np.sum(kernel)

def rotation(im): #prend une image en argument, et la ressort pivotée dans le bon sens
    [n,p]=np.shape(im)
    im2=ndimage.convolve(im,gradlap()[2])
    if n<p :
        return(rotation(rotate(im,90)))
    else:
        if im2[:,2].all()==np.zeros((n,1)).all() :
            return(rotate(im,180))
        else:
            return(im)
    
def isoler(im,X,Y) : #Renvoie la zone de l'image correspondant aux pixels entre les valeurs spécifiées par X et Y
    [px1,px2]=X    #(en proportions), utile pour tester les filtres seulement sur les zones d'interret
    [py1,py2]=Y
    [n,p]=np.shape(im)
    [x1,x2]=[int(n*px1),int(n*px2)]
    [y1,y2]=[int(p*py1),int(p*py2)]
    return(im[x1:x2,y1:y2])
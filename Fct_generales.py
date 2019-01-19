# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:12:27 2019

@author: villa
"""
from scipy import ndimage
import numpy as np


#Fonctions génerales

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